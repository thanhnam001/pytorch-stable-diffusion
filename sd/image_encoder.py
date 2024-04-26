from einops import rearrange, repeat
import torch
from torch import nn
import torchvision
from sd.transformer import PositionalEncoding, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from config import Config
class LabelStyleEncoder(nn.Module):
    def __init__(self,
                 d_model=Config.label_style_encoder.d_model,
                 nhead=Config.label_style_encoder.nhead,
                 num_encoder_layers=Config.label_style_encoder.num_encoder_layers,
                 num_head_layers=Config.label_style_encoder.num_head_layers,
                 dec_layers=Config.label_style_encoder.dec_layers,
                 dim_feedforward=Config.label_style_encoder.dim_feedforward,
                 dropout=Config.label_style_encoder.dropout,
                 activation=Config.label_style_encoder.activation,
                 normalize_before=Config.label_style_encoder.normalize_before,
                 num_writers=Config.label_style_encoder.num_writers,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Feat_Encoder = nn.Sequential(*([
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)] 
                                            +list(torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).children())[1:-2]))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.image_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)
        label_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.label_encoder = TransformerEncoder(encoder_layer, num_head_layers, label_norm)
        self.add_position = PositionalEncoding(dropout=0.1, dim=d_model) # require (L, B, C)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, dec_layers, decoder_norm)
        self.ids_embed = nn.Embedding(num_writers, d_model)
        self.label_embed = nn.Embedding(53, d_model)
        
    def forward(self, labels, style_imgs=None, writer_ids=None):
        # style_imgs: (B, C, H, W)
        # labels: (B, L, C)
        # (B, C, H, W) -> (B, C, H/32, W/32)
        labels_embed = self.label_embed(labels)
        labels = rearrange(labels_embed, 'b l c -> l b c')
        labels_positioned_feature = self.add_position(labels)
        labels_feature = self.label_encoder(labels_positioned_feature)
        
        if style_imgs is not None: # style imgs
            imgs_feature_embed = self.Feat_Encoder(style_imgs)
            imgs_feature_embed = rearrange(imgs_feature_embed, 'b c h w -> (h w) b c')
            imgs_positioned_feature_embed = self.add_position(imgs_feature_embed)
            # (L, B, C)
            imgs_feature = self.image_encoder(imgs_positioned_feature_embed) 
            if writer_ids is not None: # style + label
                # (B,) -> (B, C)
                writer_embed = self.ids_embed(writer_ids)
                imgs_feature = imgs_feature + writer_embed
        elif writer_ids is not None: # labels only
            # (B,) -> (B, C)
            imgs_feature = self.ids_embed(writer_ids)
            imgs_feature = repeat(imgs_feature, 'b c -> l b c', l=16)
        else: # no style + label
            return labels_feature
        output = self.decoder(labels_feature, imgs_feature)
        output = output[-1] # last output only
        return output

# model = LabelStyleEncoder()
# labels = torch.randint(52,(1, 10))
# imgs = None #torch.rand(1, 3, 64, 256)
# writer_ids = torch.randint(10, (1,))
# output = model(labels, imgs, writer_ids)
# print(output.shape)