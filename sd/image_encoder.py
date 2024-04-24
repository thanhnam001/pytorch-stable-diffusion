from einops import rearrange
import torch
from torch import nn
import torchvision
from transformer import PositionalEncoding, TransformerEncoder, TransformerEncoderLayer

class StyleEncoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=2,
                 num_head_layers=1,
                 wri_dec_layers=2,
                 gly_dec_layers=2,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=True,
                 return_intermediate_dec=True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Feat_Encoder = nn.Sequential(*([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)] 
                                            +list(torchvision.models.resnet18(pretrained=True).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        # writer_norm = nn.LayerNorm(d_model) if normalize_before else None
        # glyph_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.writer_head = TransformerEncoder(encoder_layer, num_head_layers, writer_norm)
        # self.glyph_head = TransformerEncoder(encoder_layer, num_head_layers, glyph_norm)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model)        
        
    def forward(self, style_imgs):
        # (B, C, H, W) -> (B, C, H/32, W/32)
        feature_embed = self.Feat_Encoder(style_imgs)
        feature_embed = rearrange(feature_embed, 'b c h w -> (h w) b c')
        positioned_feature_embed = self.add_position(feature_embed)

        transformer_out = self.base_encoder(positioned_feature_embed)
        return transformer_out

model = StyleEncoder()
input = torch.rand(1, 3, 64, 256)
output = model(input)
print(output.shape)