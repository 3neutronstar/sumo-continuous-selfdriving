import torch
import gym
from Agent.gymAgent import RLlibGymLearner
class RLLibImplementor:
    def __init__(self,flags,device,configs,sumoBinary,sumoConfigs):
        self.device=device
        self.configs=configs
        self.sumoBinary=sumoBinary
        self.sumoConfigs=sumoConfigs

    def run(self):
        if 'gym' in self.configs:
            gym_learner=RLlibGymLearner(self.configs)
            gym_learner.run()
        elif 'test' in self.configs:
            self._test()
        elif 'train' in self.configs:
            self._train()

    def _test(self):
        return

    def _train(self):
        return