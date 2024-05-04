import argparse
import os

from PIL import Image
from einops import repeat
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from dataloader.dataset import IAMDataset, Collate, WriterIdConverter
from sampler.ddpm import DDPMSampler
from sd.full_model import FullModel
from tqdm import tqdm
from sd.pipeline import timestep_embedding
from utils import *

def generate_image(text: torch.Tensor, ref_images, styles: torch.Tensor, model: FullModel, sampler: DDPMSampler) -> torch.Tensor:
    latent = torch.randn((text.shape[0],4,8,32), device=Config.device)
    context = model.label_style_encoder(text, ref_images, styles)
    for ts in tqdm(sampler.timesteps):
        ts = torch.tensor([ts])
        latent_input = latent
        time_embedding = timestep_embedding(ts, Config.timestep_embedding_dim).to(Config.device)
        model_output = model.diffuser(latent_input, context, time_embedding)
        latent = sampler.step(ts, latent, model_output)
    images = model.vae_decoder(latent)
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    images = images.to("cpu", torch.uint8)
    # (Batch_Size, Channel, Height, Width)
    return images

def train(model: FullModel,
          sampler: DDPMSampler,
          dataloader: DataLoader,
          criterion,
          optimizer: torch.optim.Optimizer,
          ):
    print('Start training')
    texts = ['text', 'getting', 'prop']
    texts_tensor = label_converter(texts, to_tensor=True) # N_text x (1, L)
    id_converter = WriterIdConverter()
    for epoch in range(1,Config.epochs+1):
        model.train()    
        print('Epoch: ', epoch)
        pbar = tqdm(dataloader)
        
        for i, (writer_ids, images, labels) in enumerate(pbar):
            writer_ids = writer_ids.to(Config.device) # (B, )
            images = images.to(Config.device) # (B, C, H, W)
            labels = labels.to(Config.device) # (B, L)

            latents_shape = (images.shape[0], 4, 8, 32) # latent shape in SD
            encoder_noise = torch.randn(latents_shape, device=Config.device)
            timesteps = sampler.timestep_sampling(images).to(Config.device)
            # sinusoidal timestep representation
            time_embedding = timestep_embedding(timesteps, Config.timestep_embedding_dim)

            latent = model.vae_encoder(images, encoder_noise)
            noisy_latent, noise = sampler.add_noise(latent, timesteps)
            # time_embedding: (B, 320)
            # encoder_noise: (B, 4, 8, 32)
            # latent: (B, 4, 8, 32)
            # context (B, L: 10, C: 512)
            # context = model.label_encoder(labels) # CLIP
            context = model.label_style_encoder(labels, images, writer_ids)
            latent_out = model.diffuser(noisy_latent, context, time_embedding)
            
            loss = criterion(latent_out, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                print('{:*^100}'.format('Generate sample'))
                for i, text_ts in enumerate(texts_tensor):
                    text_ts = text_ts.to(Config.device)
                    styles = torch.randint(0, 339, (16,)).to(Config.device)
                    dup_text_ts = repeat(text_ts,'t l -> (b t) l', b=16)
                    images = generate_image(dup_text_ts, styles, model, sampler)
                    wid = [id_converter.get_writer_id(i.item()) for i in styles]
                    save_images(images, wid, os.path.join(Config.save_path, f'images/{texts[i]}_{epoch}.jpg'))
            torch.save(model.state_dict(), os.path.join(Config.save_path, f'models/ckpt_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(Config.save_path, f'models/optimizer_{epoch}.pt'))
            print('{:*^100}'.format('Continue training'))
            
def main():
    setup_experiment()
    set_seed(Config.seed)
    
    train_ds = IAMDataset(
        root=Config.dataset_root,
        label_path=Config.label_path
        # root='C:/Users/thanh/Python/WordStylist/data',
        # label_path='gt/train_samples'
    )
    collate_fn = Collate()
    train_loader = DataLoader(train_ds, Config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = FullModel().to(Config.device)
     # load pretrain for VAE encoder decoder
    model._load_custom_pretrain(
        ckpt_path=Config.pretrained_sd,
        use_pretrained_encoder=Config.use_pretrained_encoder,
        use_pretrained_diffuser=Config.use_pretrained_diffuser,
        use_pretrained_decoder=Config.use_pretrained_decoder,
        use_pretrained_text_encoder=Config.use_pretrained_text_encoder
        )
    model.vae_encoder.requires_grad_(False)
    model.vae_decoder.requires_grad_(False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    
    generator = torch.Generator(device=Config.device)
    sampler = DDPMSampler(generator, device=Config.device)
    sampler.set_inference_timesteps(Config.inference_timestep)
    
    train(model, sampler, train_loader, criterion, optimizer)

if __name__=='__main__':
    main()