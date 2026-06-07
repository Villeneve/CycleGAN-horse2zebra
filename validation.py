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
from src.config import *

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
device = torch.device('cuda:0')
model = Generator().to(device)
model.load_state_dict(torch.load('./weights/genCut.pth'))

# %%
with torch.inference_mode():
    horses,_ = next(iter(loader))
    horses = horses[0:25].to(device)
    output = model(horses).permute(0,2,3,1).cpu()
    output = output*127.5+127.5
    output = np.uint8(output)
    fig,ax = plt.subplots(5,5,figsize=(8,8))
    ax = ax.ravel()
    for i in range(25):
        ax[i].imshow(output[i])
        ax[i].axis(False)
    plt.tight_layout(pad=0)
    plt.show()
    