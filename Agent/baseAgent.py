import torch
import os


class MainAgent():
    def __init__(self, file_path, time_data, device, configs):
        self.configs = configs
        #TODO 상속형 Cross 선언
        if configs['network'] == "cross":
            from Agent.crossAgent import CrossAgent
            self.network = CrossAgent(file_path, time_data, device, configs)
        elif configs['network'] == "grid":
            from Agent.gridAgent import GridAgent
            self.network = GridAgent(file_path, time_data, device, configs)


class BaseAgent():
    def __init__(self, file_path, time_data, device, configs):
        self.file_path = file_path
        self.time_data = time_data
        self.action_size = configs['EXP_CONFIGS']['action_size']
        self.state_size = configs['EXP_CONFIGS']['state_space']
        self.max_steps=float(configs['EXP_CONFIGS']['max_steps'])

        self.mode=configs['mode']


    def get_action(self, state, num_agent):
        raise NotImplementedError

    def update(self, epoch):
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
        raise NotImplementedError
    def save_weight(self,epoch,best_reward,total_reward):
        raise NotImplementedError

    def load_weight(self,time_data):
        raise NotImplementedError
