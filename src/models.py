import torch
import torch.nn as nn
from src.layers import *

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
            # NoiseInject2D(256),
            nn.Conv2d(256,128,3,1,1),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            # 128x128
            # NoiseInject2D(128),
            nn.Conv2d(128,64,3,1,1),
            nn.LeakyReLU(),
            nn.Conv2d(64,3,7,1,3,padding_mode='reflect'),
            nn.Tanh()
        )
        self.autoencoder = nn.Sequential(
            self.encoder,
            self.latent,
            self.decoder
        )
    
    def forward(self, x:torch.Tensor):
        return self.autoencoder(x)
    
    def features(self, x:torch.Tensor):
        feats = []
        for i,layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1,3,5]:
                feats.append(x)
        return feats
    
    def encoder_forward(self,x):
        return self.encoder(x)
    
    def decoder_forward(self,x):
        return self.decoder(x)
    
class Critic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = nn.Sequential(
            # 128x128
            ResConv2D(3,32,depth_wise=False).spectral_norm(),
            ResConv2D(32,32,depth_wise=False).spectral_norm(),
            nn.utils.spectral_norm(nn.Conv2d(32,32,4,2,1)),
            # 64x64
            ResConv2D(32,64,depth_wise=False).spectral_norm(),
            ResConv2D(64,64,depth_wise=False).spectral_norm(),
            nn.utils.spectral_norm(nn.Conv2d(64,64,4,2,1)),
            # 32x32
            ResConv2D(64,128,depth_wise=False).spectral_norm(),
            ResConv2D(128,128,depth_wise=False).spectral_norm(),
            nn.utils.spectral_norm(nn.Conv2d(128,128,4,2,1)),
            # 16x16
            nn.utils.spectral_norm(nn.Conv2d(128,1,1,1,0))
        )

    def forward(self, x: torch.Tensor):
        return self.features(x)