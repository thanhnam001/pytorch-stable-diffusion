import argparse
import os

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from dataset import IAMDataset, Collate, LabelConverter
from sd.ddpm import DDPMSampler
from sd.full_model import FullModel
from tqdm import tqdm
from sd.pipeline import timestep_embedding
from utils import *

def generate_image(text: torch.Tensor, model:FullModel, sampler) -> torch.Tensor:
    latent = torch.randn((text.shape[0],4,8,32), device=Config.device)
    context = model.label_encoder(text)
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

def train(model:nn.Module,
          sampler: DDPMSampler,
          dataloader,
          criterion,
          optimizer: torch.optim.Optimizer,
          args):
    model.train()    
    print('Start training')
    texts = ['text', 'getting', 'prop']
    texts_tensor = label_converter(texts, to_tensor=True)
    for epoch in range(1,Config.epochs+1):
        print('Epoch: ', epoch)
        pbar = tqdm(dataloader)
        
        for i, (writer_ids, images, labels) in enumerate(pbar):
            # writer_ids.to(args.device)
            images = images.to(Config.device)
            labels = labels.to(Config.device)

            latents_shape = (images.shape[0],4,8,32)
            encoder_noise = torch.randn(latents_shape, device=Config.device)
            timesteps = sampler.timestep_sampling(images).to(Config.device)
            time_embedding = timestep_embedding(timesteps, Config.timestep_embedding_dim)

            latent = model.vae_encoder(images, encoder_noise)
            noisy_latent, noise = sampler.add_noise(latent, timesteps)
            # time_embedding: (B, 320)
            # encoder_noise: (B, 4, 8, 32)
            # latent: (B, 4, 8, 32)
            context = model.label_encoder(labels)
            # print(noisy_latent.shape, context.shape, time_embedding.shape)
            latent_out = model.diffuser(noisy_latent, context, time_embedding)
            
            loss = criterion(latent_out, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())

        if epoch % 20 == 1:
            print('{:*^100}'.format('Generate sample'))
            for i, text_ts in enumerate(texts_tensor):
                text_ts = text_ts.to(Config.device)
                images = generate_image(text_ts, model, sampler)
                save_images(images, os.path.join(Config.save_path, f'images/{texts[i]}_{epoch}.jpg'))
            torch.save(model.state_dict(), os.path.join(Config.save_path, f'models/ckpt_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(Config.save_path, f'models/optimizer_{epoch}.pt'))
            print('{:*^100}'.format('Continue training'))
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam or other dataset') 
    parser.add_argument('--iam_path', type=str, default='/path/to/iam/images/', help='path to iam dataset (images 64x256)')
    parser.add_argument('--gt_train', type=str, default='./gt/gan.iam.tr_va.gt.filter27')
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./output/')
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default='./vae', help='path to stable diffusion')
    args = parser.parse_args()
    
    os.makedirs(Config.save_path, exist_ok=True)
    os.makedirs(os.path.join(Config.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(Config.save_path, 'images'), exist_ok=True)
    
    train_ds = IAMDataset(
        # root=Config.dataset_root,
        # label_path=Config.label_path
        root='C:/Users/thanh/Python/WordStylist/data',
        label_path='gt/train_samples'
    )
    collate_fn = Collate()
    train_loader = DataLoader(train_ds, Config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = FullModel(device=Config.device)
    # model._load_custom_pretrain(Config.pretrained_sd)
    model.vae_encoder.requires_grad_(False)
    model.vae_decoder.requires_grad_(False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.diffuser.parameters(), lr=Config.lr)
    
    generator = torch.Generator(device=Config.device)
    sampler = DDPMSampler(generator, device=Config.device)
    sampler.set_inference_timesteps(Config.inference_timestep)
    
    train(model, sampler, train_loader, criterion, optimizer, args)

if __name__=='__main__':
    main()