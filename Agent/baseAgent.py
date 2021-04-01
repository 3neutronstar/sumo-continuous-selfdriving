import torch
import torch.nn as nn
import torch.nn.functional as f
import traci
class MainAgent():
    def __init__(self,file_path,configs):
        self.configs =configs
        if configs['agent'] == "cross":
            from Agent.cross import CrossAgent
            self.network = CrossAgent(file_path,configs)
        elif configs['agent'] == "grid":
            from Network.grid import GridNetwork
            self.network = GridNetwork(file_path,configs)

class BaseAgent():
    def __init__(self,file_path,configs):
        self.file_path=file_path
        self.net_configs=configs['agent_configs']

        from Agent.Algorithm.dqn import DQN
        from Agent.Algorithm.ddpg import DDPG
        self.dqn_model=DQN(configs)
        self.ddpg_model=DDPG(configs)

        
    
    def get_action(self,state):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

