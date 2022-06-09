from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import glob
import torch

import numpy as np


class SvdDataset(Dataset):
    """Flowers dataset."""

    def __init__(self,
                 files_list,
                 labels_list,
                 transform=transforms.ToTensor(),
                 files_list_svd=None):
        """
        Args:
            files_list (string): list of image files.
            labels_list: list of image labels
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_list = files_list
        self.labels_list = labels_list
        self.labels = []
        self.class_labels_dict = {}
        self.transform = transform
        self.files_list_svd = files_list_svd
        
        labels_ordered = sorted(list(set(self.labels_list)))
        for i, label in enumerate(labels_ordered):
            self.class_labels_dict[label] = i
            
        for i, file in enumerate(self.files_list):
            self.labels.append(self.class_labels_dict[self.labels_list[i]])

        assert len(self.files_list) == len(self.labels)      

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        images = Image.open(self.files_list[idx])
        labels = self.labels[idx]
        
        if self.transform:
            images  = self.transform(images)

        # sample = {'image': image, 'labels': labels}

        return images, labels
    

if __name__ == '__main__':
    
    path_data = 'data/flowers.npy'
    batch_size = 256
    workers = 16
    image_size = 224
    
    triples = np.load(path_data)
    print('triples: ')
    print(triples[:10])
    print(triples[-10:])
    transform_compose=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               # transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    dataset = SvdDataset(files_list=triples[:,0], labels_list=triples[:, 1], transform=transform_compose, files_list_svd=triples[:, 0])
    print(f'len(dataset): {len(dataset)}')
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    for i, batch in enumerate(dataloader):
        # images, labels = batch['image'], batch['labels']
        images, labels = batch
        print(i, images.shape, labels.shape)