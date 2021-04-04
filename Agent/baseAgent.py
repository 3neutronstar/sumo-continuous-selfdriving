import torch
import torch.nn as nn
import torch.nn.functional as f
import traci


class MainAgent():
    def __init__(self, file_path, configs):
        self.configs = configs
        if configs['agent'] == "cross":
            from Agent.cross import CrossAgent
            self.network = CrossAgent(file_path, configs)
        elif configs['agent'] == "grid":
            from Network.grid import GridNetwork
            self.network = GridNetwork(file_path, configs)


class BaseAgent():
    def __init__(self, file_path, configs):
        self.file_path = file_path
        self.action_size = configs['EXP_CONFIGS']['action_size']
        self.state_size = configs['EXP_CONFIGS']['state_size']

        from Agent.Algorithm.dqn import DQN
        from Agent.Algorithm.ddpg import DDPG
        self.dqn_model = DQN(
            self.state_size, self.action_size, configs['AGENT_CONFIGS']['dqn'])
        self.ddpg_model = DDPG(
            self.state_size+1, self.action_size, configs['AGENT_CONFIGS']['ddpg'])

    def get_action(self, state):
        raise NotImplementedError

    def update(self, epoch):
        raise NotImplementedError
