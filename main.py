#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tt
from src.utils import FolderLoad
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import random

from src.layers import *
from src.models import *
from src.utils import *

random.seed(42)
torch.random.manual_seed(42)
torch.cuda.random.manual_seed_all(42)
np.random.default_rng(42)


#%%
compose = tt.Compose([
    tt.Resize(128,),
    tt.CenterCrop((128,128)),
    tt.ToTensor(),
    tt.Normalize((.5,.5,.5),(.5,.5,.5))
])
batch_size = 25
load_data = FolderLoad(transform=compose)
loader = DataLoader(
    load_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

#%%––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
rtx = torch.device('cuda:1')
genAB = Generator().to(rtx)
genBA = Generator().to(rtx)
# genAB = Unet().to(rtx)
# genBA = Unet().to(rtx)
cricA = Critic().to(rtx)
cricB = Critic().to(rtx)
optG = [
    torch.optim.Adam(genAB.parameters(),1e-4,betas=(0.,.99)),
    torch.optim.Adam(genBA.parameters(),1e-4,betas=(0.,.99)),
]
optC = [
    torch.optim.Adam(cricA.parameters(),1e-4,betas=(0.,.99)),
    torch.optim.Adam(cricB.parameters(),1e-4,betas=(0.,.99)),
]
BCE = nn.functional.binary_cross_entropy_with_logits
# output = genAB(torch.randn(1,3,128,128,device=rtx))

#%%
epochGraph = tqdm(range(1000),position=0,leave=True)
for epoch in epochGraph:
    batchGraph = tqdm(loader,position=1,leave=False)
    plotResult(genAB,loader)
    if epoch%10==0:
        torch.save(genAB.state_dict(),'weights/wGenAB.weight.pth')
        torch.save(genBA.state_dict(),'weights/wGenBA.weight.pth')
    genAB.train(); genBA.train(); cricA.train(); cricB.train()
    for horses,zebras in batchGraph:

        horses,zebras = horses.to(rtx),zebras.to(rtx)

        # Treinamento Generators
        fake_zebra = genAB(horses)
        fake_horse = genBA(zebras)

        fake_zebra_logits = cricB(fake_zebra)
        fake_horse_logits = cricA(fake_horse)

        rec_zebra = genAB(fake_horse)
        rec_horse = genBA(fake_zebra)

        identA = (genAB(zebras)-zebras).abs().mean()
        identB = (genBA(horses)-horses).abs().mean()

        cycleA = (horses-rec_horse).abs().mean()
        cycleB = (zebras-rec_zebra).abs().mean()

        # AdvLoss = BCE(fake_horse_logits,torch.ones_like(fake_horse_logits)).mean() + BCE(fake_zebra_logits,torch.ones_like(fake_zebra_logits)).mean()
        AdvLoss = (fake_horse_logits-1).square().mean() + (fake_zebra_logits-1).square().mean()

        optG[0].zero_grad()
        optG[1].zero_grad()
        lossG = 1*(identA+identB) + 10*(cycleA+cycleB) + AdvLoss
        lossG.backward()
        optG[0].step()
        optG[1].step()

        # Treinamento Critic
        true_horse = cricA(horses)
        fake_horse = cricA(fake_horse.detach())
        true_zebra = cricB(zebras)
        fake_zebra = cricB(fake_zebra.detach())
        # AdvA = BCE(true_horse,torch.ones_like(true_horse)).mean() + BCE(fake_horse,torch.zeros_like(fake_horse)).mean()
        # AdvB = BCE(true_zebra,torch.ones_like(true_zebra)).mean() + BCE(fake_zebra,torch.zeros_like(fake_zebra)).mean()
        AdvA = (true_horse-1).square().mean() + fake_horse.square().mean()
        AdvB = (true_zebra-1).square().mean() + fake_zebra.square().mean()
        optC[0].zero_grad()
        optC[1].zero_grad()
        lossC = AdvA + AdvB
        lossC.backward()
        optC[0].step()
        optC[1].step()

        dict_losses = {
            'Gen':f'{lossG.item():.4f}',
            'Ident':f'{5*(identA.item()+identB.item()):.4f}',
            'Cycle':f'{10*(cycleA.item()+cycleB.item()):.4f}',
            'Adv':f'{AdvLoss.item():.4f}',
            'Crit':f'{lossC.item():.4f}',
        }
        batchGraph.set_postfix(dict_losses)