import torch.nn as nn
import torch.nn.functional as f
import torch

class DDPG(nn.modules):
    def __init__(self,configs):
        self.output_size=configs['ddpg_output_size']

    def forward(self,input):
        x=input
        return x
