import os
import ray
from ray.tune.registry import register_env
import torch
import gym
from Agent.gymAgent import RLlibGymLearner
import libsumo as traci
class RLLibImplementor:
    def __init__(self,flags,device,configs,sumoBinary,sumoConfig):
        self.device=device
        self.configs=configs
        self.sumoBinary=sumoBinary
        self.sumoConfig=sumoConfig
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.file_path=file_path

    def run(self):
        if 'gym' in self.configs['mode']:
            gym_learner=RLlibGymLearner(self.configs)
            gym_learner.run()
        elif 'test' in self.configs['mode']:
            self._test()
        elif 'train' in self.configs['mode']:
            self._train()

    def _test(self):
        return

    def _train(self):
        STEP_LENGTH=self.configs['step_length']
        sumoCmd = [self.sumoBinary, "-c", self.sumoConfig,"--step-length",str(STEP_LENGTH)]
        from Env.baseEnv_rllib import Env
        from ray.rllib.agents import dqn

        def env_creator(env_config):
            return Env(env_config['file_path'],env_config['device'],env_config['configs'])
        register_env("sumoenv",env_creator)
        ###
        ## pip install tensorflow, tensorflow-gpu
        ## ray 확인
        ## pip install libsumo
        ## https://github.com/lcodeca/rllibsumoutils ## installation 할때, git clone하고 거기 폴더 안에서 명령어 install 있는거 치셈 python3 setup.py install
        ## 우리 명령어: python main.py train_rllib
        ## pip install lz4
        ## cv2 오류는 pip install opencv-python
        ## tree 오류는 pip install dm-tree

        ray.init()
        trainer=dqn.DQNTrainer(env='sumoenv',config={'env':Env,'env_config':{
                'file_path':self.file_path,
                'device':self.device,
                'configs':self.configs
            },
            'lr':1e-4,
            'num_workers':1,
            'framework':'torch',
            'horizon':3000,
            })
        for e in range(1,self.configs['epochs']+1):
            trainer.train()#안됨
        return
        # 목표1: 연결시키기 (sumo랑 ray랑)
        # 목표2: for e in range()돌아가는게 맞는지
        # 목표3: horizon없으면 epoch이 안끝나는걸로 아는데 맞는지 확인해주세요
        # 돌아가면 야호를 외쳐주세요
