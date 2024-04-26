from PIL import Image, ImageDraw,ImageFont
import cv2
import os
import torch
import torchvision
from dataloader.dataset import LabelConverter

from config import Config

def print_config(f, obj,indent=4):
    attrs = vars(obj)
    f.write(' ' * (indent - 4) + obj.__name__ + ':\n')
    for k, v in attrs.items():
        if (k.startswith('__')): continue
        if type(v)!=type:
            f.write(' '*indent + f'{k}: {v}\n')
        else:
            print_config(f, v, indent + 4)
            
def setup_experiment():
    i = 0
    while True:
        if not os.path.exists(f'{Config.save_path}_{i}'):
            Config.save_path = f'{Config.save_path}_{i}'
            break
        i += 1
    print('Current experiment path: ', Config.save_path)
    os.makedirs(Config.save_path, exist_ok=True)
    os.makedirs(os.path.join(Config.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(Config.save_path, 'images'), exist_ok=True)
    with open(f'{Config.save_path}/config.yaml','w') as f:
        print_config(f, Config, 4)

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

def save_images(images, labels, path, latent=Config.latent, **kwargs):
    h, w = Config.img_size
    rows, cols = 2, 8
    background = Image.new('RGB', size=(cols*w, rows*h))
    if latent == True:
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, (image, label) in enumerate(zip(images, labels)):
            im = torchvision.transforms.ToPILImage()(image)
            # print(im.shape,label.item())
            
            # im.show()
            draw = ImageDraw.Draw(im)
            font = ImageFont.truetype('arial.ttf',size=16)
            draw.text((0,0),label,(255,0,0),font=font)
            background.paste(im, box=(i%cols*w, i//cols*h))
    else:
        pass
        # ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        # im = Image.fromarray(ndarr)
    background.save(path)
    return background