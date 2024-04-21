import torch
from torch import nn
from torch.nn import functional as F

from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.diffusion import Diffusion
from sd import model_converter

class FullModel(nn.Module):
    def __init__(self, device: str='cpu', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device
        self.vae_encoder = VAE_Encoder().to(device)
        self.vae_decoder = VAE_Decoder().to(device)
        self.diffuser = Diffusion().to(device)
        
    def _load_custom_pretrain(self, 
                              ckpt_path,
                              use_pretrained_encoder=True,
                              use_pretrained_diffuser=False,
                              use_pretrained_decoder=True,
    ):
        state_dict = model_converter.load_from_standard_weights(ckpt_path, self.device)
        if use_pretrained_encoder:
            self.vae_encoder.load_state_dict(state_dict['encoder'], strict=True)
        if use_pretrained_decoder:
            self.vae_decoder.load_state_dict(state_dict['decoder'], strict=True)
        if use_pretrained_diffuser:
            self.diffuser.load_state_dict(state_dict['diffusion'], strict=True)
    
    def forward(self, batch, timesteps):
        raise NotImplementedError