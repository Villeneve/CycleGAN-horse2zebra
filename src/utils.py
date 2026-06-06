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
    num_patches: int = 256,
    temperature: float = 0.07,
    detach_key: bool = True,
):
    """
    Calcula PatchNCE entre dois mapas de features.

    feat_q: mapa query, geralmente features de fake_B
            shape: (B, C, H, W)

    feat_k: mapa key, geralmente features de real_A
            shape: (B, C, H, W)

    Retorna:
        loss escalar
    """

    if feat_q.shape != feat_k.shape:
        raise ValueError(f"feat_q e feat_k devem ter o mesmo shape. "
                         f"Recebido {feat_q.shape} e {feat_k.shape}")

    B, C, H, W = feat_q.shape
    N = H * W

    # (B, C, H, W) -> (B, H*W, C)
    feat_q = feat_q.permute(0, 2, 3, 1).reshape(B, N, C)
    feat_k = feat_k.permute(0, 2, 3, 1).reshape(B, N, C)

    # Amostra patches correspondentes
    if num_patches is not None and num_patches < N:
        patch_ids = torch.randperm(N, device=feat_q.device)[:num_patches]
        feat_q = feat_q[:, patch_ids, :]
        feat_k = feat_k[:, patch_ids, :]

    if detach_key:
        feat_k = feat_k.detach()

    # Normaliza no eixo dos canais
    feat_q = F.normalize(feat_q, dim=2, eps=1e-8)
    feat_k = F.normalize(feat_k, dim=2, eps=1e-8)

    # Depois da amostragem
    S = feat_q.shape[1]

    # Positivo: q_i com k_i
    # shape: (B, S, 1)
    l_pos = torch.bmm(
        feat_q.view(B * S, 1, C),
        feat_k.view(B * S, C, 1)
    ).view(B, S, 1)

    # Negativos: q_i com todos os k_j da mesma imagem
    # shape: (B, S, S)
    l_neg = torch.bmm(
        feat_q,
        feat_k.transpose(1, 2)
    )

    # Remove o positivo da matriz de negativos
    diagonal = torch.eye(S, device=feat_q.device, dtype=torch.bool)[None, :, :]
    l_neg = l_neg.masked_fill(diagonal, -10.0)

    # Logits finais: positivo na coluna 0, negativos depois
    # shape: (B, S, 1 + S)
    logits = torch.cat([l_pos, l_neg], dim=2)

    # Temperatura
    logits = logits / temperature

    # Cada patch deve escolher a coluna 0, que é o positivo
    labels = torch.zeros(B * S, dtype=torch.long, device=feat_q.device)

    # Cross entropy espera (N, classes)
    logits = logits.view(B * S, 1 + S)

    loss = F.cross_entropy(logits, labels)

    return loss

def lr_lambda(epoch: int) -> float:
    # epoch é a época atual (começa em 0)
    # retorna um MULTIPLICADOR, não o LR diretamente
    if epoch < 100:
        return 1.0                        # LR = 2e-4 × 1.0 = 2e-4 (sem mudança)
    else:
        return 1.0 - (epoch - 100) / 900 # decai linearmente até 0 ao fim de 1000 épocas