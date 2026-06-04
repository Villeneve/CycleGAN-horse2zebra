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
class ResBatchConv2D(nn.Module):
    def __init__(self, inCh, outCh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual = nn.Sequential(
            nn.Conv2d(inCh,outCh,3,1,1),
            nn.BatchNorm2d(outCh),
            nn.LeakyReLU(),
            nn.Conv2d(outCh,outCh,3,1,1,groups=outCh),
            nn.BatchNorm2d(outCh),
        )
        self.skip = nn.Identity() if inCh == outCh else nn.Conv2d(inCh,outCh,1,1,0)
        self.gamma = nn.Parameter(torch.ones(1,outCh,1,1)*.1)
    
    def forward(self, x: torch.Tensor):
        skip = self.skip(x)
        residual = self.residual(x)
        return skip + self.gamma*residual
    
class ResConv2D(nn.Module):
    def __init__(self, inCh, outCh, depth_wise=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if depth_wise:
            self.residual = nn.Sequential(
                nn.Conv2d(inCh,outCh,3,1,1),
                nn.LeakyReLU(),
                nn.Conv2d(outCh,outCh,3,1,1,groups=outCh),
                nn.LeakyReLU(),
                nn.Conv2d(outCh,outCh,1,1,0)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(inCh,outCh,3,1,1),
                nn.LeakyReLU(),
                nn.Conv2d(outCh,outCh,3,1,1),
            )
        for layer in self.residual:
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,0.01)
                nn.init.zeros_(layer.bias)
        self.skip = nn.Identity() if inCh == outCh else nn.Conv2d(inCh,outCh,1,1,0)
        self.gamma = nn.Parameter(torch.ones(1,outCh,1,1)*.2)
    
    def forward(self, x: torch.Tensor):
        skip = self.skip(x)
        residual = self.residual(x)
        return skip + self.gamma*residual
    
class NoiseInject2D(nn.Module):
    def __init__(self, inCh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inCh = inCh
        self.gamma = nn.Parameter(.01*torch.ones(1,inCh,1,1))

    def forward(self, x: torch.Tensor):
        B,_,H,W = x.size()
        noise = self.gamma*torch.randn(B,1,H,W,device=x.get_device())
        return x+noise
    
class Unet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            # 128x128
            ResConv2D(3,32,False),
            ResConv2D(32,32,False),
            nn.AvgPool2d(2,2),
            # 64x64
            ResConv2D(32,64,False),
            ResConv2D(64,64,False),
            nn.AvgPool2d(2,2),
            # 32x32
            ResConv2D(64,128,False),
            ResConv2D(128,128,False),
            nn.AvgPool2d(2,2),            
        )
        self.latent = nn.Sequential(
            # 16x16
            ResConv2D(128,256,False),
            ResConv2D(256,256,False),
            ResConv2D(256,256,False),
            ResConv2D(256,256,False),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.decoder = nn.Sequential(
            # 32x32
            ResConv2D(256+128,128,False),
            ResConv2D(128,128,False),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 64x64
            ResConv2D(128+64,64,False),
            ResConv2D(64,64,False),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 128x128
            ResConv2D(64+32,32,False),
            ResConv2D(32,32,False),
            nn.Conv2d(32,3,3,1,1),
        )

    def forward(self, x:torch.Tensor):
        skip128 = self.encoder[:2](x)
        skip64 = self.encoder[2:5](skip128)
        skip32 = self.encoder[5:8](skip64)
        latent = self.encoder[8:](skip32)
        latent = self.latent(latent)
        skip32 = self.decoder[:3](torch.cat([skip32,latent],dim=1))
        skip64 = self.decoder[3:6](torch.cat([skip64,skip32],dim=1))
        skip128 = self.decoder[6:](torch.cat([skip128,skip64],dim=1))
        return nn.functional.tanh(skip128)
    
class UnetSum(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        return

class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            # 128x128
            nn.Conv2d(3,64,7,1,3,padding_mode='reflect'),
            nn.LeakyReLU(),

            nn.Conv2d(64,128,3,2,1),
            nn.LeakyReLU(),
            # 64x64
            nn.Conv2d(128,256,3,2,1),
            nn.LeakyReLU(),
            # 32x32
        )
        self.latent = nn.Sequential(
            # 32x32
            *[ResConv2D(256,256,depth_wise=False) for i in range(6)],
            # 32x32
        )
        self.decoder = nn.Sequential(
            # 32x32
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 64x64
            NoiseInject2D(256),
            nn.Conv2d(256,128,3,1,1),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 128x128
            NoiseInject2D(128),
            nn.Conv2d(128,64,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(64,3,7,1,3),
            nn.Tanh()
        )
        self.autoencoder = nn.Sequential(
            self.encoder,
            self.latent,
            self.decoder
        )
    
    def forward(self, x:torch.Tensor):
        return self.autoencoder(x)
    
    def encoder_forward(self,x):
        return self.encoder(x)
    
    def decoder_forward(self,x):
        return self.decoder(x)
    
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = nn.Sequential(
            # 128x128
            ResConv2D(3,32,depth_wise=False),
            ResConv2D(32,32,depth_wise=False),
            nn.AvgPool2d(2,2),
            # 64x64
            ResConv2D(32,64,depth_wise=False),
            ResConv2D(64,64,depth_wise=False),
            nn.AvgPool2d(2,2),
            # 32x32
            ResConv2D(64,128,depth_wise=False),
            ResConv2D(128,128,depth_wise=False),
            nn.AvgPool2d(2,2),
            # 16x16
            ResConv2D(128,256,depth_wise=False),
            ResConv2D(256,256,depth_wise=False),
            nn.AvgPool2d(2,2),
            # 8x8
            ResConv2D(256,256,False),
            ResConv2D(256,256,False),
            nn.AvgPool2d(2,2),
            # 4x4
            nn.Conv2d(256,1,1,1,0)
        )

    def forward(self, x: torch.Tensor):
        return self.features(x)
    
def toImage(x:torch.Tensor):
    x = x.detach().permute(1,2,0).cpu().numpy()
    x *= 127.5
    x += 127.5
    x = x.astype(np.uint8)
    return x

def plotResult():
    with torch.no_grad():
        genAB.eval()
        horses,_ = next(iter(loader))
        horses = horses.to(rtx)
        fake_horses = genAB(horses)
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
    genAB.train()

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
    plotResult()
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
        lossG = 5*(identA+identB) + 10*(cycleA+cycleB) + AdvLoss
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