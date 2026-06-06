import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, inCh, outCh, kernel=3, stride=1, padding=0,**kwargs):
        super().__init__(**kwargs)
        self.cnn = nn.Conv2d(inCh,outCh,kernel,stride,padding,**kwargs)
        self.lrelu = nn.LeakyReLU(inplace=True)
        nn.init.kaiming_normal_(self.cnn.weight,.01)
        nn.init.zeros_(self.cnn.bias)
        
    def forward(self,x):
        return self.lrelu(self.cnn(x))
    
class ResConv2D(nn.Module):
    def __init__(self, inCh, outCh, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inCh,self.outCh = inCh, outCh

        self.residual = nn.Sequential(
            nn.ReflectionPad2d(1),
            ConvBlock(inCh,outCh,3,1,0),
            nn.ReflectionPad2d(1),
            nn.Conv2d(outCh,outCh,3,1,0),
        )
        nn.init.xavier_normal_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        
        if inCh == outCh:
            self.skip = nn.Identity() 
        else:
            self.skip = nn.Conv2d(inCh,outCh,1,1,0)
            nn.init.xavier_normal_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

        self.gamma = nn.Parameter(torch.ones(1,outCh,1,1)*.2)
    
    def forward(self, x: torch.Tensor):
        skip = self.skip(x)
        residual = self.residual(x)
        return skip + residual*self.gamma
        
class NoiseInject2D(nn.Module):
    def __init__(self, inCh, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inCh = inCh
        self.gamma = nn.Parameter(.01*torch.ones(1,inCh,1,1))

    def forward(self, x: torch.Tensor):
        B,_,H,W = x.size()
        noise = self.gamma*torch.randn(B,1,H,W,device=x.device)
        return x+noise