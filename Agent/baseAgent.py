import torch
import torch.nn as nn
import torch.nn.functional as f
import traci


class MainAgent():
    def __init__(self, file_path, configs):
        self.configs = configs
        if configs['network'] == "cross":
            from Agent.crossAgent import CrossAgent
            self.network = CrossAgent(file_path, configs)
        elif configs['network'] == "grid":
            from Agent.gridAgent import GridAgent
            self.network = GridAgent(file_path, configs)


class BaseAgent():
    def __init__(self, file_path, configs):
        self.file_path = file_path
        self.action_size = configs['EXP_CONFIGS']['action_size']
        self.state_size = configs['EXP_CONFIGS']['state_space']

        from Agent.Algorithm.dqn import DQN
        from Agent.Algorithm.ddpg import DDPG
        
        self.dqn_model = DQN(
            self.state_size, self.action_size, configs['AGENT_CONFIGS']['dqn'])
        self.ddpg_model = DDPG(
            self.state_size+1, self.action_size, configs['AGENT_CONFIGS']['ddpg'])

    def get_action(self, state, num_agent):
        raise NotImplementedError

    def update(self, epoch):
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('dqn/epsilon', self.dqn_model.epsilon, epoch)
        writer.add_scalar(
            'dqn/lr', self.dqn_model.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar(
            'ddpg/actor_lr', self.ddpg_model.actor_optim.param_groups[0]['lr'], epoch)
        writer.add_scalar(
            'ddpg/critic_lr', self.ddpg_model.actor_optim.param_groups[0]['lr'], epoch)
