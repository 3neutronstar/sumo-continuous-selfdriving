import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from Agent.Algorithm.utils import ReplayMemory
from Agent.Algorithm.random_process import OrnsteinUhlenbeckProcess
from Agent.Algorithm.utils import hard_update, soft_update
from collections import namedtuple
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(Actor, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.output_size = output_size
        self.fc = self._make_layers()
        self.init_weights(3e-3)


    def forward(self, input):
        x = input
        x = self.fc(x)
        x = F.tanh(x)
        return x

    def _make_layers(self):
        layers = []
        fc_list = [self.input_size]+self.configs['fc']

        for i, fc in enumerate(fc_list):
            if fc_list[i] == fc_list[-1]:
                layers += [nn.Linear(fc, self.output_size),nn.Tanh()]
                break
            layers += [nn.Linear(fc, fc_list[i+1]),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def init_weights(self, init_w):
        num_fc=0
        for fc in self.fc:
            if isinstance(fc,nn.Linear):
                num_fc+=1
                if num_fc ==(len(self.configs['fc'])):
                    fc.weight.data=fanin_init(fc.weight.data.size())
                else:
                    fc.weight.data.uniform_(-init_w, init_w)
            
class Critic(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(Critic, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.output_size = output_size
        self.fc = self._make_layers()
        self.init_weights(3e-3)

    def forward(self, input, actions):
        x = torch.cat((input, actions), dim=1)
        x = self.fc(x)
        return x

    def init_weights(self, init_w):
        num_fc=0
        for fc in self.fc:
            if isinstance(fc,nn.Linear):
                num_fc+=1
                if num_fc ==(len(self.configs['fc'])):
                    fc.weight.data=fanin_init(fc.weight.data.size())
                else:
                    fc.weight.data.uniform_(-init_w, init_w)

    def _make_layers(self):
        layers = []
        fc_list = [self.input_size+1]+self.configs['fc']

        for i, fc in enumerate(fc_list):
            if fc_list[i] == fc_list[-1]:
                layers += [nn.Linear(fc, self.output_size)]
                break
            layers += [nn.Linear(fc, fc_list[i+1]),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


class DDPG():
    def __init__(self, input_size, output_size, mode,device, configs):
        self.configs = configs
        self.device = device
        self.mode=mode 
        
        self.gamma = configs['gamma']
        self.actor = Actor(input_size, output_size, configs['actor'])
        self.actor.to(self.device)
        self.actor_target = Actor(input_size, output_size, configs['critic'])
        self.actor_target.to(self.device)
        self.actor_optim = optim.Adam(
            self.actor.parameters(), configs['actor']['lr'],weight_decay=5e-4)
        self.actor_lr_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optim, step_size=configs['actor']['lr_decaying_epoch'], gamma=configs['actor']['lr_decaying_rate'])
        hard_update(self.actor_target, self.actor)

        self.critic = Critic(input_size, output_size, configs['critic'])
        self.critic.to(self.device)
        self.critic_target = Critic(input_size, output_size, configs['critic'])
        self.critic_target.to(self.device)
        self.critic_optim = optim.Adam(
            self.critic.parameters(), configs['critic']['lr'],weight_decay=5e-4)
        self.critic_lr_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optim, step_size=configs['critic']['lr_decaying_epoch'], gamma=configs['critic']['lr_decaying_rate'])
        hard_update(self.critic_target, self.critic)

        self.action_noise = OrnsteinUhlenbeckProcess(
            size=output_size, theta=configs['ou']['theta'], mu=configs['ou']['mu'], sigma=configs['ou']['sigma'])
        self.noise_scale=self.configs['initial_noise_scale']

        self.action_top = self.configs['action_space'][1]
        self.action_down = self.configs['action_space'][0]


        if self.configs['gym_mode']==False:
            self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward', 'next_state'))
        else:
            self.Transition = namedtuple('Transition',
                                    ('state', 'action', 'reward', 'next_state','done'))
        self.experience_replay = ReplayMemory(
            configs['experience_replay_size'],self.Transition)
        self.actor_target.eval()
        self.critic_target.eval()

    def get_action(self, state):
        if self.mode=='train' and self.configs['init_train_ddpg']>=len(self.experience_replay):
            mu=torch.randn([state.size()[0],1]).to(self.device)
        else:
            with torch.no_grad():
                self.actor_target.eval()
                mu = self.actor_target(state.float().to(self.device))
                mu = mu.data*self.action_top

                if self.action_noise is not None and self.mode !='test':
                    noise = torch.Tensor(self.action_noise.sample()*self.noise_scale).to(
                        self.device)
                    mu += noise
        mu = mu.clamp(self.action_down, self.action_top)
        return mu

    def update(self, next_action=None, epoch=None):
        self.actor.train()
        self.critic.train()
        if len(self.experience_replay) <= self.configs['init_train_ddpg']:
            return 0.0, 0.0
        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = self.Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(
            batch.next_state).to(self.device)
        if self.configs['gym_mode']==False:
            state_batch = torch.cat((state_batch, action_batch.detach().clone()), dim=1)
            next_state_batch = torch.cat(
                (next_state_batch, next_action), dim=1)
        else:
            done_batch=torch.tensor(batch.done).to(self.device)
        
        # print('{} {} {} {} {}'.format(state_batch.sum(),action_batch.sum(),reward_batch.sum(),next_state_batch.sum(),done_batch.sum()))

        # get action and the state value from each target
        next_action_batch = self.actor_target(next_state_batch).detach()

        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch).detach()

        # calc target
        reward_batch = reward_batch.unsqueeze(1)
        if self.configs['gym_mode']==False:
            expected_values = reward_batch + self.gamma*next_state_action_values
        else:
            done_batch = done_batch.unsqueeze(1)
            expected_values = reward_batch + (~done_batch) * self.gamma * next_state_action_values

        # critic network update
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.smooth_l1_loss(
            state_action_batch, expected_values)
        value_loss.backward()
        self.critic_optim.step()

        # actor network update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss=policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # update target
        soft_update(self.actor_target, self.actor,
                    self.configs['actor']['tau'])
        soft_update(self.critic_target, self.critic,
                    self.configs['critic']['tau'])

        return value_loss.item(), policy_loss.item()

    def save_replay(self, state, action, reward, next_state, done=None):
        # print(state, action, reward, next_state)
        # 0 index인 이유는 dqn과 섞이기 때문
        if done==None:
            self.experience_replay.push(
                state, action[:, 0].view(1, -1), reward, next_state)
        else:
            self.experience_replay.push(
                state, action[:, 0].view(1, -1), reward, next_state, done)


    def hyperparams_update(self):
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        if self.noise_scale<self.configs['final_noise_scale']:
            self.noise_scale=self.configs['final_noise_scale']
        else:
            self.noise_scale*=self.configs['noise_reduce_rate']
    
    def eval(self):
        self.actor_target.eval()
        self.critic_target.eval()