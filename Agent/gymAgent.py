import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
class GymLearner():
    def __init__(self,flags,device,configs):
        self.device=device
        self.configs=configs
        self.algorithm=flags.algorithm

    def run(self):
        if self.algorithm=='ddpg':
            self.run_continuous()
        elif self.algorithm=='dqn':
            self.run_discrete()

    def run_continuous(self):
        number_of_episodes = 300

        step_size_initial = 1
        step_size_decay = 1

        # INITIALIZATION

        evol = []
        env = gym.make('Pendulum-v0')
        self.configs['action_space'] = env.action_space
        self.configs['state_size'] = 3
        add_dict={
        'actor': {'fc': [200, 100, 100], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.05},
        'critic': {'fc': [200, 100, 100], 'lr': 1e-3, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.05},
        'experience_replay_size': 1e5,
        'batch_size': 64,
        'ou': {'theta': 0.15, 'sigma': 0.2, 'mu': 0.0},
        'gamma': 0.99,
        'action_space': [-1.0, 1.0],'init_train_ddpg':3000,
        'gym_mode':True,
        'initial_noise_scale':1.0,
        'final_noise_scale':0.01,
        'noise_reduce_rate':0.99,
        }
        self.configs=dict(self.configs,**add_dict)
        print(env.action_space)
        print(env.observation_space)
        step_size = step_size_initial
        if self.algorithm=='ddpg':
            from Agent.Algorithm.ddpg import DDPG
        else:
            raise NotImplementedError
        learner = DDPG(self.configs['state_size'],1,self.configs['mode'],self.device,self.configs)
        t = 0
        reward = 0
        writer=SummaryWriter('./training_data/gym/')
        for e in range(number_of_episodes):
            state = env.reset()
            t = 0
            done = False
            state = torch.from_numpy(state).reshape(1, -1).float()
            Return = 0
            while not done:
                t += 1
                if e >=298:
                    env.render()
                action = learner.get_action(state)

                next_state, reward, done, _ = env.step(action.cpu())
                next_state = torch.from_numpy(np.array(next_state,dtype=np.float32)).reshape(1, -1)
                # print("{} {} {} {} {}".format(state, action, reward, next_state,done))
                learner.save_replay(state, action, torch.tensor([reward],device=self.device), next_state, done)
                loss = learner.update(next_action=None,epoch=e)
                state = next_state
                Return += reward
            # learner.update_hyperparams(e)
            # if e % 10 == 0:
            #     learner.target_update()

            writer.add_scalar('value_loss',loss[0],e)
            writer.add_scalar('policy_loss',loss[1],e)
            writer.add_scalar('reward',Return,e)
            writer.flush()
            print('Episode ' + str(e) + ' ended in ' +
                str(t) + ' time steps, reward: ', str(Return))
            print("loss: {}".format(loss))

class RLlibGymLearner:
    def __init__(self,configs):
        self.configs=configs
    
    def run(self):
        if self.configs['algorithm']=='ddpg' or self.configs['algorithm']=='ppo':
            self._continuous_run()
        elif self.configs['algorithm']=='dqn':
            self._discrete_run()

    def _continuous_run(self):
        import ray
        from ray import tune
        from ray.rllib.agents import ppo,ddpg
        ray.init(num_cpus=4,num_gpus=1,local_mode=True)
        configs={
            'num_gpus':1,
            'num_workers':4,
            # 'num_gpus_per_worker':1,
            'framework':'torch',
            "simple_optimizer":True,
        }
        AGENT_CONFIG={
            'ddpg':ddpg.DDPGTrainer(config=configs,env="MountainCarContinuous-v0"),
            'ppo':ppo.PPOTrainer(config=configs,env="MountainCarContinuous-v0"),
        }
        trainer=AGENT_CONFIG[self.configs['algorithm']]
        # tune.run(agent, config={"env": "MountainCarContinuous-v0","framework":"torch","num_gpus":0,})
        for i in range(2000): # 2000epoch
            result=trainer.train()#1 epoch
            print(result)

        return

    def _discrete_run(self):

        from ray import tune
        from ray.rllib.agents import dqn
        from ray.rllib.agents.dqn import DEFAULT_CONFIG
        DEFAULT_CONFIG['framework']='torch'
        if self.configs['mode']:
            DEFAULT_CONFIG['double_q']=False
        else:
            DEFAULT_CONFIG['double_q']=True
        import ray
        AGENT_CONFIG={'dqn':dqn.DQNTrainer,
        'ddqn':dqn.DQNTrainer(config=DEFAULT_CONFIG,env='CartPole-v0')}
        agent=AGENT_CONFIG[self.configs['algorithm']]
        tune.run(agent, config={"env": "CartPole-v0","framework":"torch"})
