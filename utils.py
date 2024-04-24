from PIL import Image

import torch
import torchvision
from dataset import LabelConverter

from config import Config

def rescale(x, old_range, new_range, clamp=False):
    '''
    Rescale image from `old_range` to `new_range` and clamp if needed
    '''
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def label_converter(texts: list[str], converter=LabelConverter, to_tensor=True):
    cvt = converter()
    wrapper = lambda x: torch.tensor([x]) if to_tensor else lambda x: x
    converted_ids = []
    for text in texts:
        ids = cvt.string_to_ids(text)
        ids = wrapper(ids)
        converted_ids.append(ids)
    return converted_ids

def save_images(images, path, latent=Config.latent, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im