import argparse

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

def train(model:FullModel,
          sampler: DDPMSampler, dataloader, criterion, optimizer, args):
    model.train()    
    print('Start training')
    latents_shape = (3,4,8,32)
    for epoch in range(args.epochs):
        print('Epoch: ', epoch)
        pbar = tqdm(dataloader)
        
        for i, (writer_ids, images, labels) in enumerate(pbar):
            # writer_ids.to(args.device)
            images.to(args.device)
            labels.to(args.device)
            
            encoder_noise = torch.randn(latents_shape, device=args.device)
            latent = model.vae_encoder(images, encoder_noise)
            timesteps = sampler.timestep_sampling(images)
            time_embedding = timestep_embedding(timesteps, 320)
            # time_embedding: (B, 320)
            # encoder_noise: (B, 4, 8, 32)
            # latent: (B, 4, 8, 32)
            context = 
            latent_out = model.diffuser(latent, context, time_embedding)
            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=3)
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
    parser.add_argument('--save_path', type=str, default='./save_path/')
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default='./vae', help='path to stable diffusion')
    args = parser.parse_args()
    
    train_ds = IAMDataset(
        root='data',
        label_path='gt/gan.iam.tr_va.gt.filter27'
    )
    collate_fn = Collate()
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = FullModel()
    model.vae_encoder.requires_grad_(False)
    model.vae_decoder.requires_grad_(False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.diffuser.parameters(), lr=0.0001)
    
    generator = torch.Generator()
    sampler = DDPMSampler(generator)
    
    train(model, sampler, train_loader, criterion, optimizer, args)

if __name__=='__main__':
    main()