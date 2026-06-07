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
batch_size = 8
lr = 2e-4
beta1 = .5
beta2 = .999

nce_temperature = .07

device = torch.device('cuda:1')

#%%
transform = tt.Compose([
    tt.Resize(img_size+16,tt.InterpolationMode.BILINEAR),
    tt.RandomCrop((img_size,img_size)),
    tt.RandomHorizontalFlip(.5),
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
netF = PatchProjector([64, 128, 256, 256, 256, 256], proj_dim=256).to(device)

summary(gen, (batch_size, 3, img_size, img_size), verbose=1)
summary(crit, (batch_size, 3, img_size, img_size), verbose=1)

optG = torch.optim.Adam(
    list(gen.parameters()) + list(netF.parameters()),
    lr,
    betas=(beta1, beta2)
)

optC = torch.optim.Adam(
    crit.parameters(),
    lr,
    betas=(beta1, beta2)
)

scheduler_G = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda=lr_lambda)
scheduler_C = torch.optim.lr_scheduler.LambdaLR(optC, lr_lambda=lr_lambda)

#%%
# epochGraph = tqdm(range(10),position=0)
for epoch in range(1000+1):
    batchGraph = tqdm(loader,position=0)
    if epoch%10 == 0: torch.save(gen.state_dict(),'./weights/genCut.pth')
    for horses,zebras in batchGraph:
        batchGraph.set_description(f'Epoch: {epoch}')
        horses,zebras = horses.to(device),zebras.to(device)

        # Forward Crítico
        setGrads(crit,True)
        with torch.no_grad():
            fakeZebras = gen(horses)
        trueLogits = crit(zebras)
        fakeLogits = crit(fakeZebras)
        AdvCritLoss = 0.5*((trueLogits-1).square().mean() + fakeLogits.square().mean())
        optC.zero_grad()
        AdvCritLoss.backward()
        optC.step()

        # Forward Gerador
        setGrads(crit, False)
        optG.zero_grad()

        fakeZebras = gen(horses)

        fakeLogits = crit(fakeZebras)
        AdvGenLoss = (fakeLogits - 1).square().mean()

        featLoss_X = multilayer_patch_nce_loss(
            gen.features(fakeZebras),
            gen.features(horses),
            netF,
            num_patches=256,
            temperature=nce_temperature
        )

        idtZebras = gen(zebras)

        featLoss_Y = multilayer_patch_nce_loss(
            gen.features(idtZebras),
            gen.features(zebras),
            netF,
            num_patches=256,
            temperature=nce_temperature
        )

        lossG = AdvGenLoss + featLoss_X + featLoss_Y
        lossG.backward()
        optG.step()

        # Plot Loss
        dictLoss = {
            'featLoss_X': f'{featLoss_X.item():.4f}',
            'featLoss_Y': f'{featLoss_Y.item():.4f}',
            'AdvGenLoss': f'{AdvGenLoss.item():.4f}',
            'AdvCritLoss': f'{AdvCritLoss.item():.4f}'
        }
        batchGraph.set_postfix(dictLoss)
    plotResult(gen,loader)
    # scheduler_G.step()
    # scheduler_C.step()
    # epochGraph.set_postfix(dictLoss)

#%%
with torch.inference_mode():
    horse,_ = next(iter(loader))
    horse = horse[0:1].to(device)
    fakeZebra = gen(horse)
    fakeZebra = fakeZebra[0].permute(1,2,0).cpu().numpy()*127.5+127.5
    fakeZebra = fakeZebra.astype(np.uint8)
    plt.imshow(fakeZebra)
    plt.axis(False)
    plt.show()