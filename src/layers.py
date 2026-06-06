import torch.nn as nn
import torch

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
    def __init__(self, inCh, outCh, depth_wise=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inCh,self.outCh = inCh, outCh
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
                nn.ReflectionPad2d(1),
                nn.Conv2d(inCh,outCh,3,1,0),
                nn.InstanceNorm2d(outCh),
                nn.LeakyReLU(),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(outCh,outCh,3,1,0),
                nn.InstanceNorm2d(outCh),
            )
        for layer in self.residual:
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,0.01)
                nn.init.zeros_(layer.bias)
        self.skip = nn.Identity() if inCh == outCh else nn.Conv2d(inCh,outCh,1,1,0)
        # self.gamma = nn.Parameter(torch.ones(1,outCh,1,1)*.2)
    
    def forward(self, x: torch.Tensor):
        skip = self.skip(x)
        residual = self.residual(x)
        return skip + residual
    
    def spectral_norm(self):
        self.residual = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(self.inCh,self.outCh,3,1,1)),
                nn.LeakyReLU(),
                nn.utils.spectral_norm(nn.Conv2d(self.outCh,self.outCh,3,1,1)),
            )
        if isinstance(self.skip,nn.Conv2d):
            self.skip = nn.utils.spectral_norm(nn.Conv2d(self.inCh,self.outCh,1,1,0))
        return self
        
    
class NoiseInject2D(nn.Module):
    def __init__(self, inCh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inCh = inCh
        self.gamma = nn.Parameter(.01*torch.ones(1,inCh,1,1))

    def forward(self, x: torch.Tensor):
        B,_,H,W = x.size()
        noise = self.gamma*torch.randn(B,1,H,W,device=x.get_device())
        return x+noise