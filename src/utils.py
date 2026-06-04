from torch.utils.data import Dataset
from PIL import Image
import os
import random

class FolderLoad(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.horses = os.listdir('/storage/SSD1/.data/horse2zebra/trainA')
        self.zebras = os.listdir('/storage/SSD1/.data/horse2zebra/trainB')
        self.horses = [os.path.join('/storage/SSD1/.data/horse2zebra/trainA',i) for i in self.horses]
        self.zebras = [os.path.join('/storage/SSD1/.data/horse2zebra/trainB',i) for i in self.zebras]

    def __getitem__(self, index):
        horse = Image.open(self.horses[index]).convert('RGB')
        zebra = Image.open(self.zebras[random.randint(0,len(self.zebras)-1)]).convert('RGB')
        if self.transform is not None:
            horse = self.transform(horse)
            zebra = self.transform(zebra)
        return horse,zebra

    def __len__(self):
        return len(self.horses)