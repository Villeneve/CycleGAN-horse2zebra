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

def setGrads(model:nn.Module,value:bool):
    for p in model.parameters():
        p.requires_grad_(value)

import torch.nn.functional as F

def patch_nce_loss(
    feat_q: torch.Tensor,
    feat_k: torch.Tensor,
    temperature: float = 0.07,
    detach_key: bool = True,
):
    if detach_key:
        feat_k = feat_k.detach()

    B, S, C = feat_q.shape

    l_pos = torch.bmm(
        feat_q.reshape(B * S, 1, C),
        feat_k.reshape(B * S, C, 1)
    ).reshape(B, S, 1)

    l_neg = torch.bmm(
        feat_q,
        feat_k.transpose(1, 2)
    )

    diagonal = torch.eye(S, device=feat_q.device, dtype=torch.bool)[None, :, :]
    l_neg = l_neg.masked_fill(diagonal, -10.0)

    logits = torch.cat([l_pos, l_neg], dim=2)
    logits = logits / temperature
    logits = logits.reshape(B * S, 1 + S)

    labels = torch.zeros(B * S, dtype=torch.long, device=feat_q.device)

    return F.cross_entropy(logits, labels)

def lr_lambda(epoch: int) -> float:
    # epoch é a época atual (começa em 0)
    # retorna um MULTIPLICADOR, não o LR diretamente
    if epoch < 100:
        return 1.0                        # LR = 2e-4 × 1.0 = 2e-4 (sem mudança)
    else:
        return 1.0 - (epoch - 100) / 900 # decai linearmente até 0 ao fim de 1000 épocas
    
def multilayer_patch_nce_loss(
    feats_q,
    feats_k,
    netF,
    num_patches=256,
    temperature=0.07,
):
    feats_q, patch_ids = netF(
        feats_q,
        patch_ids=None,
        num_patches=num_patches
    )

    feats_k, _ = netF(
        feats_k,
        patch_ids=patch_ids,
        num_patches=num_patches
    )

    total_loss = 0.0

    for feat_q, feat_k in zip(feats_q, feats_k):
        total_loss = total_loss + patch_nce_loss(
            feat_q,
            feat_k,
            temperature=temperature
        )

    return total_loss / len(feats_q)