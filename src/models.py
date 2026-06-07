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
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 128x128
            nn.ReflectionPad2d(1),
            ConvBlock(3,64,3,1,0),
            nn.ReflectionPad2d(1),
            ConvBlock(64,128,4,2,0),
            # 64x64
            nn.ReflectionPad2d(1),
            ConvBlock(128,256,4,2,0),
            # 32x32
        )
        self.latent = nn.Sequential(
            # 32x32
            *[ResConv2D(256,256) for i in range(6)],
            # 32x32
        )
        self.decoder = nn.Sequential(
            # 32x32
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            ConvBlock(128,128,3,1,1),
            # 64x64
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64,64,3,1,1),
            # 128x128
            nn.Conv2d(64,3,7,1,3,padding_mode='reflect'),
            nn.Tanh()
        )
        for layer in self.decoder:
            if isinstance(layer,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight,.01)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.decoder[-2].weight)
        nn.init.zeros_(self.decoder[-2].bias)

        self.autoencoder = nn.Sequential(
            self.encoder,
            self.latent,
            self.decoder
        )
    
    def forward(self, x:torch.Tensor):
        return self.autoencoder(x)
    
    def features(self, x: torch.Tensor):
        feats = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [1, 3, 5]:
                feats.append(x)

        for i, layer in enumerate(self.latent):
            x = layer(x)
            if i in [1, 3, 5]:
                feats.append(x)

        return feats
    
    def encoder_forward(self,x):
        return self.encoder(x)
    
    def decoder_forward(self,x):
        return self.decoder(x)
    
class Critic(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        def snconv(in_c, out_c, k=4, s=2, p=1):
            return (
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            )

        self.model = nn.Sequential(
            # 256x256
            snconv(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128
            snconv(base_ch, base_ch * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64
            snconv(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32
            snconv(base_ch * 4, base_ch * 8, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(base_ch * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)
    
class PatchProjector(nn.Module):
    def __init__(self, channels, proj_dim=256):
        super().__init__()

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim)
            )
            for c in channels
        ])

    def forward(self, feats, patch_ids=None, num_patches=256):
        projected_feats = []
        sampled_ids = []

        for i, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

            if patch_ids is None:
                ids = torch.randperm(H * W, device=feat.device)
                ids = ids[:min(num_patches, H * W)]
            else:
                ids = patch_ids[i]

            feat = feat[:, ids, :]
            feat = self.mlps[i](feat)
            feat = nn.functional.normalize(feat, dim=2, eps=1e-8)

            projected_feats.append(feat)
            sampled_ids.append(ids)

        return projected_feats, sampled_ids