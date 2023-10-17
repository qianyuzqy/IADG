import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from pdb import set_trace as st
import numpy as np
import math
import sys
sys.path.append("..") 
from utils.initial import init_weights
from .dkg_module import DKGModule

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    '''
        the cnn module with specific parameters
        Args:
            in_channels (int): the channel numbers of input features
            out_channels (int): the channel numbers of output features
            stride (int): the stride paramters of Conv2d
            padding (int): the padding parameters of Conv2D
            bias (bool): the bool parameters of Conv2d
    '''
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)

# New implementation of basic blocks------------------------------------------------------------
class Conv_block_gate(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dkg_flag, model_initial='kaiming'):
        '''
            Args:
                in_channels (int): the channel numbers of input features
                out_channels (int): the channel numbers of output features
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                model_initial (str):
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Conv_block_gate, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        if self.dkg_flag==True:
            self.conv = DKGModule(3, in_channels, out_channels, m=16, padding=1, stride=1)
        else:     
            self.conv = conv3x3(in_channels, out_channels)
        self.norm = nn.InstanceNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # model initial
        init_weights(self.conv, init_type=self.model_initial)

    def forward(self, x):
        # Nomal branch of conb+bn+relu
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        # print(x.shape)
            
        return out

class Basic_block_gate(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dkg_flag, model_initial='kaiming'):
        '''
            Basic_block contains three Conv_block

            Args:
                in_channels (int): the channel numbers of input features
                out_channels (int): the channel numbers of output features
                padding (int): the padding parameters of conv block
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Basic_block_gate, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        self.padding = padding
        if self.dkg_flag==True:
            self.conv_block1_gate = Conv_block_gate(in_channels, 128, 0, False, self.model_initial)
            self.conv_block2_gate = Conv_block_gate(128, 196, self.padding, True, self.model_initial)
            self.conv_block3_gate = Conv_block_gate(196, out_channels, 0, False, self.model_initial)
        else:
            self.conv_block1_gate = Conv_block_gate(in_channels, 128, 0, False, self.model_initial)
            self.conv_block2_gate = Conv_block_gate(128, 196, 0, False, self.model_initial)
            self.conv_block3_gate = Conv_block_gate(196, out_channels, 0, False, self.model_initial)   
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, input):
        # print("conv1")
        input = self.conv_block1_gate(input)
        # print("conv2")
        input = self.conv_block2_gate(input)
        # print("conv3")
        input = self.conv_block3_gate(input)
        input = self.max_pool(input)
        return input