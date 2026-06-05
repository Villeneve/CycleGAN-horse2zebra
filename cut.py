#%% Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tt
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from PIL import Image
import random

from src.utils import *
from src.layers import *
from src.models import *
from src.utils import *

#%% Hiperparâmetros
img_size = 128
batch_size = 32
lr = 2e-4
beta1 = .5
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
batchGraph = tqdm()
for horses,zebras in loader:

    horses,zebras = horses.to(device),zebras.to(device)

    # Forward Crítico
    fakeZebras = gen(horses).detach()
    trueLogits = crit(zebras)
    fakeLogits = crit(fakeZebras)
    AdvCritLoss = (trueLogits-1).square().mean() + fakeLogits.square().mean()
    optC.zero_grad()
    AdvCritLoss.backward()
    optC.step()

    # Forward Gerador
    fakeZebras = gen(horses)
    fakeLogits = crit(fakeZebras)
    AdvGenLoss = (fakeLogits-1).square().mean()
    optG.zero_grad()
    AdvGenLoss.backward()
    optG.step()


