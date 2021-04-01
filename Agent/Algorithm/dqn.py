import torch.nn as nn
import torch.nn.functional as f
import torch

class DQN(nn.modules):
    def __init__(self,configs):

        self.fc1=nn.Linear()

        self.output_size=configs['dqn_output_size']

    def forward(self,input):
        x=input
        return x