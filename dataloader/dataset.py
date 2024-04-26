import torch
from torch import nn
import numpy
import cv2
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision

from config import Config

class LabelConverter:
    def __init__(self, charset:str = Config.charset, max_seq_len: int = Config.max_seq_len) -> None:
        self.charset = charset
        self.max_seq_len = max_seq_len
        self.char_to_id = {c:i for i,c in enumerate(self.charset)}
        self.id_to_char = {i:c for i,c in enumerate(self.charset)}
        self.pad_token = len(self.charset) # see Config.charset
        
    def string_to_ids(self, string: str):
        ids = [self.char_to_id[c] for c in string]
        if len(ids) < self.max_seq_len:
            ids += [self.pad_token] * (self.max_seq_len - len(ids))
        return ids
    
    def ids_to_string(self, ids: list[int]):
        chars = [self.id_to_char.get(i, '') for i in ids]
        return ''.join(chars)

class WriterIdConverter:
    def __init__(self, root: str = Config.dataset_root, label_path: str = Config.label_path) -> None:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        self.wid_to_trainid = dict()
        train_id = 0
        for full_line in lines:
            line = full_line.strip()
            id, _ = line.split(',')
            if id not in self.wid_to_trainid.keys():
                self.wid_to_trainid[id] = train_id
                train_id += 1
        self.trainid_to_wid = {v:k for k, v in self.wid_to_trainid.items()}
    
    def get_train_id(self, index: str) -> int:
        return self.wid_to_trainid[index]
    
    def get_writer_id(self, index: int) -> str:
        return self.trainid_to_wid[index]
    
                
class IAMDataset(Dataset):
    def __init__(self, root: str, label_path: str) -> None:
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
    
    def __len__(self):
        return len(self.full_data)
    
    def __getitem__(self, index):
        sample = self.full_data[index]
        
        image = Image.open(sample['image_path']).convert('RGB')
        
        label = sample['label']
        
        writer_id = sample['writer_id']

        return writer_id, image, label

class Collate(object):
    def __init__(self, image_transform=None, label_converter=LabelConverter, wid_converter=WriterIdConverter):
        self.image_transform =  torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.label_converter = label_converter()
        self.wid_converter = wid_converter()
        
    def __call__(self, batch):
        writer_ids, images, labels = zip(*batch)
        writer_ids = [self.wid_converter.get_train_id(i) for i in writer_ids]
        writer_ids = torch.tensor(writer_ids)
        
        images = [self.image_transform(i) for i in images]
        images = torch.stack(images)
        
        new_labels = []
        for label in labels:
            label = self.label_converter.string_to_ids(label)
            new_labels.append(label)
        new_labels = torch.tensor(new_labels)
        return writer_ids, images, new_labels
        