import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
    
def toImage(x:torch.Tensor):
    x = x.detach().permute(1,2,0).cpu().numpy()
    x *= 127.5
    x += 127.5
    x = x.astype(np.uint8)
    return x

def plotResult(gen:nn.Module,loader):
    with torch.no_grad():
        gen.eval()
        horses,_ = next(iter(loader))
        horses = horses.to(next(gen.parameters()).device)
        fake_horses = gen(horses)
        imgs = torch.cat([horses,fake_horses],dim=0)
        width = imgs.size(0)
        fig,ax = plt.subplots(2,width//2,figsize=(width//2,2))
        ax = ax.ravel()
        for i in range(width):
            ax[i].imshow(toImage(imgs[i]))
            ax[i].axis(False)
        plt.tight_layout(pad=0)
        plt.savefig('result.png')
        plt.close()
    gen.train()