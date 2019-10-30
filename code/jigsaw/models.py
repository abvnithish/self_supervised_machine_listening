
### Importing Libraries
import os
import time
import random
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import librosa
from librosa.display import specshow
import scipy, IPython.display as ipd
# from utils import *

### Setting seed for reproducibility
random_state = 7
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

debug_flag=False

def debug(str):
    if debug_flag:
        print(str)

def load_model(device, model_arch , mode, separable_falg):
   
    model_name = model_arch + '_' + mode
    if separable_falg:
          model_name += '_separable'
    print(f'Instantiating Model {model_name}')

    if model_name == 'anet_jigsaw':
        model = anet_jigsaw().to(device)
    elif model_name == 'snet_flip':
        model = snet_flip().to(device)
    elif model_name == 'snet_jigsaw':
        model = snet_jigsaw().to(device)
    elif model_name == 'snet_jigsaw_separable':
        model = snet_jigsaw_separable().to(device)
    elif model_name == 'l3net_jigsaw':
        model = l3net_jigsaw().to(device)
    elif model_name == 'l3net_jigsaw_separable':
        model = l3net_jigsaw_separable().to(device)
    elif model_name == 'l3net_flip_separable':
        model = l3net_flip_separable().to(device)
    
    return model

class anet_jigsaw(nn.Module):

    def __init__(self):
        super(anet_jigsaw, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=1,     # This is fixed to 1 for raw audio input.
            out_channels=5,
            kernel_size=5,
            stride=2,
            padding=0)

        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0)

        self.dense = torch.nn.Linear(
            in_features=15,    # This must match 'out_channels' from self.conv.
            out_features=6) # This is fixed by imagenet.

    def forward(self, x):
        #x = x.unsqueeze(1) # Add a channel dimension.
        debug('Starting')
        debug(x.shape)
        conv_strips = []
        n_strips = x.shape[1]
        for strip in range(n_strips):
            temp = x[:,strip]
            temp = temp.unsqueeze(1)
            debug('Strip shape')
            debug(temp.shape)
            conv_strips.append(self.conv(temp))
        #conv_strips=torch.from_numpy(np.array(conv_strips))
        debug('CONV strips shape')
        debug(conv_strips[0].shape)
        x=torch.cat(conv_strips,1)
        debug('After CONV')
        debug(x.shape)
        x = self.maxpool(x)
        debug('After MAXPOOL')
        debug(x.shape)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=x.size()[2])
        debug('After AVGPOOL')
        debug(x.shape)
        x = x.squeeze(2) # Remove the averaged value's dimension.
        debug('before DENSE')
        debug(x.shape)
        x = x.view(x.size(0), -1)
        debug('before DENSE')
        debug(x.shape)
        x = self.dense(x)
        debug('after DENSE')
        debug(x.shape)
        return x


class snet_flip(nn.Module):

    def __init__(self):
        '''
        Create the 5 Conv Layer Sound Net network architecture as per the paper - https://arxiv.org/pdf/1610.09001.pdf
        '''
        super(snet_flip, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels= 16, kernel_size = 5, stride = 2, padding = 2), 
                    nn.BatchNorm2d(num_features = 16), 
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(kernel_size = 3, stride = (1,2)),

                    nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = 2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(kernel_size = 3, stride = (1,2)),

                    nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace = True),

                    nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True),

                    nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True),
                    nn.MaxPool2d(kernel_size = 4, stride = 1),

    #                     nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 2),
    #                     nn.BatchNorm2d(512),
    #                     nn.ReLU(inplace = True),

    #                     nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 2),
    #                     nn.BatchNorm2d(1024),
    #                     nn.ReLU(inplace = True)
                    )
        self.mlp_layer = nn.Linear(256, 2)

    def forward(self, input):

        out =  self.conv_layers(input)
        logits = self.mlp_layer(out.view(out.shape[0], -1))
        return logits


class snet_jigsaw(nn.Module):

    def __init__(self):
        '''
        Create the 5 Conv Layer Sound Net network architecture as per the paper - https://arxiv.org/pdf/1610.09001.pdf
        '''
        super(snet_jigsaw, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels= 16, kernel_size = 5, stride = 2, padding = 2), 
                nn.BatchNorm2d(num_features = 16), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 4, stride = (1,2)),

                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = (1,2)),

                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 1),

#                     nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 2),
#                     nn.BatchNorm2d(512),
#                     nn.ReLU(inplace = True),

#                     nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 2),
#                     nn.BatchNorm2d(1024),
#                     nn.ReLU(inplace = True)
                )
        self.concat_mlp_layer = nn.Linear(2304, 256)
        #self.mlp_layer = nn.Linear(256, 6)
        self.mlp_layer = nn.Linear(256, 2)
              
    def forward(self, input):
        debug('Starting')
        debug(input.shape)
        conv_strips = []
        n_strips = input.shape[1]
        for strip in range(n_strips):
            conv_strip = input[:,strip]
            conv_strip = conv_strip.unsqueeze(1)
            debug('Strip shape')
            debug(conv_strip.shape)
            conv_strips.append(self.conv_layers(conv_strip))
        #conv_strips=torch.from_numpy(np.array(conv_strips))
        debug('CONV strips shape')
        debug(conv_strips[0].shape)
        concat_out=torch.cat(conv_strips,1)
        out = self.concat_mlp_layer(concat_out.view(concat_out.shape[0], -1))
        #out =  self.conv_layers(input)
        output = self.mlp_layer(out.view(out.shape[0], -1))
        return output

              
class snet_jigsaw_separable(nn.Module):

    def __init__(self):
        '''
        Create the 5 Conv Layer Sound Net network architecture as per the paper - https://arxiv.org/pdf/1610.09001.pdf
        '''
        super(snet_jigsaw_separable, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels= 3, kernel_size = 5, stride = 2, padding = 2, groups=3),
                nn.Conv2d(in_channels = 3, out_channels= 16, kernel_size = 1, stride = 2, padding = 2),
                nn.BatchNorm2d(num_features = 16), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 4, stride = (1,2)),

                nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 2, padding = 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = (1,2)),

                nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 1),

#                     nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 2),
#                     nn.BatchNorm2d(512),
#                     nn.ReLU(inplace = True),

#                     nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 2),
#                     nn.BatchNorm2d(1024),
#                     nn.ReLU(inplace = True)
                )
        #self.concat_mlp_layer = nn.Linear(768, 256)
        #self.mlp_layer = nn.Linear(256, 6)
        self.mlp_layer = nn.Linear(256, 2)
              
    def forward(self, input):
        debug('Starting')
        debug(input.shape)
        out = self.conv_layers(input)
        #out = self.concat_mlp_layer(out.view(out.shape[0], -1))
        #out =  self.conv_layers(input)
        output = self.mlp_layer(out.view(out.shape[0], -1))
        return output


class l3net_jigsaw(nn.Module):

    def __init__(self):
        '''
        Create the L3 Net network architecture
        '''
        super(l3net_jigsaw, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels= 64, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),

                nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 128, out_channels= 128, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),

                nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 256, out_channels= 256, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                                         
                nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(512), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 512, out_channels= 512, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(512), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = (3,9), stride = 0)
                )
        self.concat_mlp_layer = nn.Linear(1536, 128)
        #self.mlp_layer = nn.Linear(128, 6)
        self.mlp_layer = nn.Linear(128, 2)
              
    def forward(self, input):
        debug('Starting')
        debug(input.shape)
        conv_strips = []
        n_strips = input.shape[1]
        for strip in range(n_strips):
            conv_strip = input[:,strip]
            conv_strip = conv_strip.unsqueeze(1)
            debug('Strip shape')
            debug(conv_strip.shape)
            conv_strips.append(self.conv_layers(conv_strip))
        #conv_strips=torch.from_numpy(np.array(conv_strips))
        debug('CONV strips shape')
        debug(conv_strips[0].shape)
        concat_out=torch.cat(conv_strips,1)
        out = self.concat_mlp_layer(concat_out.view(concat_out.shape[0], -1))
        #out =  self.conv_layers(input)
        output = self.mlp_layer(out.view(out.shape[0], -1))
        return output


class l3net_jigsaw_separable(nn.Module):

    def __init__(self):
        '''
        Create the L3 Net network architecture
        '''
        super(l3net_jigsaw_separable, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels= 3, kernel_size = 3, padding = 0,groups=3),#depthwise
                nn.Conv2d(in_channels = 3, out_channels= 64, kernel_size = 1, padding = 0),#pointwise
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),

                nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 128, out_channels= 128, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),

                nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 256, out_channels= 256, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                                         
                nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(512), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 512, out_channels= 512, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(512), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = (3,9), stride = 0)
                )
        self.concat_mlp_layer = nn.Linear(512, 128)
        #self.mlp_layer = nn.Linear(128, 6)
        self.mlp_layer = nn.Linear(128, 2)
              
    def forward(self, input):
        debug('Starting')
        debug(input.shape)
        out = self.conv_layers(input)
        out = self.concat_mlp_layer(out.view(out.shape[0], -1))
        #out =  self.conv_layers(input)
        output = self.mlp_layer(out.view(out.shape[0], -1))
        return output

class l3net_flip_separable(nn.Module):

    def __init__(self):
        '''
        Create the L3 Net network architecture
        '''
        super(l3net_flip_separable, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels= 1, kernel_size = 3, padding = 0,groups=1),#depthwise
                nn.Conv2d(in_channels = 1, out_channels= 64, kernel_size = 1, padding = 0),#pointwise
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 64, out_channels= 64, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),

                nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 128, out_channels= 128, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(128), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),

                nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 256, out_channels= 256, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(256), 
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                                         
                nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(512), 
                nn.ReLU(inplace = True),

                nn.Conv2d(in_channels = 512, out_channels= 512, kernel_size = 3, padding = 0), 
                nn.BatchNorm2d(512), 
                nn.ReLU(inplace = True),
                nn.AdaptiveMaxPool2d((1,1))
                )
        self.concat_mlp_layer = nn.Linear(512, 128)
        #self.mlp_layer = nn.Linear(128, 6)
        self.mlp_layer = nn.Linear(128, 2)
              
    def forward(self, input):
        debug('Starting')
        debug(input.shape)
        out = self.conv_layers(input)
        out = self.concat_mlp_layer(out.view(out.shape[0], -1))
        #out =  self.conv_layers(input)
        output = self.mlp_layer(out.view(out.shape[0], -1))
        return output












