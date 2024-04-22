import argparse
import os

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import IAMDataset, Collate
from sd.ddpm import DDPMSampler
from sd.diffusion import Diffusion
from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.full_model import FullModel
from tqdm import tqdm
from sd.pipeline import timestep_embedding

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def train(model:FullModel,
          sampler: DDPMSampler, dataloader, criterion, optimizer, args):
    model.train()    
    print('Start training')
    charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    char_to_id = {c:i for i,c in enumerate(charset)}
    id_to_char = {i:c for i,c in enumerate(charset)}
    label_transform = lambda x: [char_to_id[c] for c in x]
    texts = ['text', 'getting', 'prop']
    texts_tensor = []
    label_max_len = 10
    for text in texts:
        text = label_transform(text)
        if len(text) < label_max_len:
            text += (label_max_len - len(text)) * [52]
        texts_tensor.append(torch.tensor([text]))
    for epoch in range(1,args.epochs+1):
        print('Epoch: ', epoch)
        pbar = tqdm(dataloader)
        
        for i, (writer_ids, images, labels) in enumerate(pbar):
            # writer_ids.to(args.device)
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents_shape = (images.shape[0],4,8,32)
            encoder_noise = torch.randn(latents_shape, device=args.device)
            timesteps = sampler.timestep_sampling(images).to(args.device)
            time_embedding = timestep_embedding(timesteps, 320)

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

        if epoch % 20 == 0:
            for i, text in enumerate(texts_tensor):
                text = text.to(args.device)
                latent = torch.randn((1,4,8,32), device=args.device)
                context = model.label_encoder(text)
                for ts in tqdm(sampler.timesteps):
                    ts = torch.tensor([ts])
                    latent_input = latent
                    time_embedding = timestep_embedding(ts, 320).to(args.device)
                    # print(latent_input.shape, context.shape, time_embedding.shape)
                    model_output = model.diffuser(latent_input, context, time_embedding)
                    latent = sampler.step(ts, latent, model_output)
                images = model.vae_decoder(latent)
                images = rescale(images, (-1, 1), (0, 255), clamp=True)
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
                images = images.permute(0, 2, 3, 1)
                images = images.to("cpu", torch.uint8).numpy()
                Image.fromarray(images[0]).save(os.path.join(args.save_path, f'images/{texts[i]}_{epoch}.jpg'))
            torch.save(model.state_dict(), os.path.join(args.save_path, f'models/ckpt_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'models/optimizer_{epoch}.pt'))
            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
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
    parser.add_argument('--device', type=str, default='cuda:1') 
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default='./vae', help='path to stable diffusion')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)
    
    train_ds = IAMDataset(
        root='/data/ocr/namvt17/WordStylist/data',
        label_path='gt/gan.iam.tr_va.gt.filter27'
    )
    collate_fn = Collate()
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = FullModel(device=args.device)
    model._load_custom_pretrain('/data/ocr/namvt17/WordStylist/diffusers/v1-5-pruned-emaonly.ckpt')
    model.vae_encoder.requires_grad_(False)
    model.vae_decoder.requires_grad_(False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.diffuser.parameters(), lr=0.0001)
    
    generator = torch.Generator(device=args.device)
    sampler = DDPMSampler(generator, device=args.device)
    sampler.set_inference_timesteps(50)
    
    train(model, sampler, train_loader, criterion, optimizer, args)

if __name__=='__main__':
    main()