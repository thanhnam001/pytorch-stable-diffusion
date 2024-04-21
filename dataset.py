import torch
from torch import nn
import numpy
import cv2
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision

class IAMDataset(Dataset):
    def __init__(self, root: str, label_path: str, image_transform=None, label_transform=None) -> None:
        super().__init__()
        with open(label_path, 'r') as f:
            lines = f.readlines()
        self.full_data = []
        for full_line in lines:
            line = full_line.strip()
            id, remain = line.split(',')
            image, word = remain.split()
            image_path = os.path.join(root,image+'.png')
            assert os.path.exists(image_path), f'{image_path} is not exist'
            self.full_data.append({'writer_id': id, 'image_path': image_path, 'label': word})
        self.image_transform = image_transform
        self.label_transform = label_transform
    
    def __len__(self):
        return len(self.full_data)
    
    def __getitem__(self, index):
        sample = self.full_data[index]
        
        image = Image.open(sample['image_path']).convert('RGB')
        # image = self.image_transform(image)
        
        # label = self.label_transform(sample['label'])
        label = sample['label']
        
        writer_id = int(sample['writer_id'])

        return writer_id, image, label

class Collate(object):
    def __init__(self, image_transform=None, label_transform=None):
        self.charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.image_transform =  torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.char_to_id = {c:i for i,c in enumerate(self.charset)}
        self.id_to_char = {i:c for i,c in enumerate(self.charset)}
        self.label_transform = lambda x: [self.char_to_id[c] for c in x]
        self.label_max_len = 10
        self.pad_token = 52

    def __call__(self, batch):
        writer_ids, images, labels = zip(*batch)
        writer_ids = torch.tensor(writer_ids)
        
        images = [self.image_transform(i) for i in images]
        images = torch.stack(images)
        
        new_labels = []
        for label in labels:
            label = self.label_transform(label)
            if len(label) < self.label_max_len:
                label += (self.label_max_len - len(label)) * [self.pad_token]
            new_labels.append(label)
        new_labels = torch.tensor(new_labels)
        return writer_ids, images, new_labels
        