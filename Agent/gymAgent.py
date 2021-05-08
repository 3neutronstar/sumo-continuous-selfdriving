import gym
import torch
import numpy as np
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
        'actor': {'fc': [200, 100, 100], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.005},
        'critic': {'fc': [200, 100, 100], 'lr': 1e-3, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.005},
        'experience_replay_size': 1e5,
        'batch_size': 64,
        'ou': {'theta': 0.15, 'sigma': 0.2, 'mu': 0.0},
        'gamma': 0.99,
        'action_space': [-1.0, 1.0],'init_train_ddpg':3000,
        'gym_mode':True,
        }
        self.configs=dict(self.configs,**add_dict)
        print(env.action_space)
        print(env.observation_space)
        step_size = step_size_initial
        if self.algorithm=='ddpg':
            from Agent.Algorithm.ddpg import DDPG
        else:
            raise NotImplementedError
        learner = DDPG(self.configs['state_size'],1,self.device,self.configs)
        t = 0
        reward = 0
        for e in range(number_of_episodes):
            state = env.reset()
            t = 0
            done = False
            state = torch.from_numpy(state).reshape(1, -1).float()
            Return = 0
            while not done:
                t += 1
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

            print('Episode ' + str(e) + ' ended in ' +
                str(t) + ' time steps, reward: ', str(Return))
            print("loss: {}".format(loss))
