import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from sd.clip import CLIP
from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.diffusion import Diffusion
from sd import model_converter
from sd.image_encoder import LabelStyleEncoder

class FullModel(nn.Module):
    def __init__(self,
                 vocab_size: int=53,
                 max_seq_len: int=Config.max_seq_len,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vae_encoder = VAE_Encoder()
        self.vae_decoder = VAE_Decoder()
        self.diffuser = Diffusion()
        self.label_encoder = CLIP(vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.label_style_encoder = LabelStyleEncoder()
        
    def _load_custom_pretrain(self, 
                              ckpt_path,
                              use_pretrained_encoder=True,
                              use_pretrained_diffuser=False,
                              use_pretrained_decoder=True,
                              use_pretrained_text_encoder=False,
    ):
        if not (use_pretrained_encoder or 
                use_pretrained_diffuser or 
                use_pretrained_decoder or
                use_pretrained_text_encoder):
            print('No pretrained modules in use')
            return
        state_dict = model_converter.load_from_standard_weights(ckpt_path, Config.device)
        if use_pretrained_encoder:
            print('Use pretrained VAE encoder')
            self.vae_encoder.load_state_dict(state_dict['encoder'], strict=True)
        if use_pretrained_decoder:
            print('Use pretrained VAE decoder')
            self.vae_decoder.load_state_dict(state_dict['decoder'], strict=True)
        if use_pretrained_diffuser:
            print('Use pretrained diffuser')
            self.diffuser.load_state_dict(state_dict['diffusion'], strict=True)
        if use_pretrained_text_encoder:
            print('Use pretrained text encoder CLIP')
            self.label_encoder.load_state_dict(state_dict['clip'], strict=True)
    
    def forward(self, batch, timesteps):
        raise NotImplementedError