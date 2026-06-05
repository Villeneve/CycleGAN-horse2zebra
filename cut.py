#%% Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from torchvision import transforms as tt
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import os
from PIL import Image
import random

from src.utils import *
from src.layers import *
from src.models import *
from src.utils import *

#%% Hiperparâmetros
img_size = 128
batch_size = 25
lr = 1e-4
beta1 = 0.
beta2 = .99

nce_temperature = .07

device = torch.device('cuda:0')

#%%
transform = tt.Compose([
    tt.Resize(128),
    tt.CenterCrop((128,128)),
    tt.ToTensor(),
    tt.Normalize((.5,.5,.5),(.5,.5,.5)),
])
loader = DataLoader(
    FolderLoad(transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

#%%
gen = Generator().to(device)
crit = Critic().to(device)
summary(gen,(32,3,128,128),verbose=1)
summary(crit,(32,3,128,128),verbose=1)
optG = torch.optim.Adam(gen.parameters(),lr,betas=(beta1,beta2))
optC = torch.optim.Adam(crit.parameters(),lr,betas=(beta1,beta2))

#%%
# epochGraph = tqdm(range(10),position=0)
for epoch in range(10):
    batchGraph = tqdm(loader,position=0)
    for horses,zebras in batchGraph:
        batchGraph.set_description(f'Epoch: {epoch}')
        horses,zebras = horses.to(device),zebras.to(device)

        # Forward Crítico
        fakeZebras = gen(horses).detach()
        trueLogits = crit(zebras)
        fakeLogits = crit(fakeZebras)
        AdvCritLoss = 0.5*((trueLogits-1).square().mean() + fakeLogits.square().mean())
        optC.zero_grad()
        AdvCritLoss.backward()
        optC.step()

        # Forward Gerador
        optG.zero_grad()
        featLoss = 0
        fakeZebras = gen(horses)
        for trueFeat,fakeFeat in zip(gen.features(horses),gen.features(fakeZebras)):
            featLoss += (normalize(trueFeat.detach(),dim=1,eps=1e-8)-normalize(fakeFeat,dim=1,eps=1e-8)).abs().mean()/3
        featLoss.backward()
        idtZebras = gen(zebras)
        idtLoss = (zebras-idtZebras).abs().mean()
        idtLoss.backward()
        fakeZebras = gen(horses)
        fakeLogits = crit(fakeZebras)
        AdvGenLoss = (fakeLogits-1).square().mean()
        AdvGenLoss.backward()

        lossG = idtLoss+AdvGenLoss+featLoss
        # lossG.backward()
        optG.step()

        # Plot Loss
        dictLoss = {
            'featLoss':f'{featLoss.item():.4f}',
            'idtLoss':f'{idtLoss.item():.4f}',
            'AdvGenLoss':f'{AdvGenLoss.item():.4f}',
            'AdvCritLoss':f'{AdvCritLoss.item():.4f}'
        }
        batchGraph.set_postfix(dictLoss)

    # epochGraph.set_postfix(dictLoss)

#%%
with torch.inference_mode():
    horse,_ = next(iter(loader))
    horse = horse[0:1].to(device)
    fakeZebra = gen(horse)
    fakeZebra = fakeZebra[0].permute(1,2,0).cpu().numpy()*127.5+127.5
    fakeZebra = fakeZebra.astype(np.uint8)
    plt.imshow(fakeZebra)
    plt.show()