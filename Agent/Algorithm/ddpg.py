import torch.nn as nn
import torch.nn.functional as f
import torch
import torch.optim as optim
from Agent.Algorithm.utils import ReplayMemory, Transition
from Agent.Algorithm.random_process import OrnsteinUhlenbeckProcess
from Agent.Algorithm.utils import hard_update, soft_update


class Actor(nn.modules):
    def __init__(self, input_size, output_size, configs):
        super(Actor, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.output_size = output_size
        self.fc = self._make_layers()

    def forward(self, input):
        x = input
        x = self.fc(x)
        return x

    def _make_layers(self):
        layers = []
        fc_list = [self.input_size]+self.configs['fc']

        for i, fc in enumerate(fc_list):
            if fc_list[i] == fc_list[-1]:
                layers += [nn.Linear(fc, self.output_size)]
                break
            layers += [nn.Linear(fc, fc_list[i+1]),
                       nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


class Critic(nn.modules):
    def __init__(self, input_size, output_size, configs):
        super(Critic, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.output_size = output_size
        self.before_critic, self.after_critic = self._make_layers()

    def forward(self, input, actions):
        x = input
        x = self.before_critic(x)
        x = torch.cat((x, actions), dim=1)
        x = self.after_critic(x)
        return x

    def _make_layers(self):
        layers = []
        fc_list = [self.input_size]+self.configs['fc']

        for i, fc in enumerate(fc_list):
            if fc_list[i] == fc_list[-1]:
                layers += [nn.Linear(fc, self.output_size)]
                after_critic = nn.Sequential(*layers)
                break
            if i+1 == len(fc_list)/2:
                layers += [nn.Linear(fc+1,),
                           nn.ReLU(inplace=True)]
                before_critic = nn.Sequential(*layers)
                layers = []
            layers += [nn.Linear(fc),
                       nn.ReLU(inplace=True)]
        return before_critic, after_critic


class DDPG():
    def __init__(self, input_size, output_size, configs):
        self.configs = configs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = configs['gamma']
        self.actor = Actor(input_size, output_size, configs)
        self.actor.to(self.device)
        self.actor_target = Actor(input_size, output_size, configs)
        self.actor_target.to(self.device)
        self.actor_optim = optim.Adam(
            self.actor.parameters(), configs['actor']['lr'])
        self.actor_lr_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optim, step_size=configs['actor']['lr_decaying_epoch'], gamma=configs['actor']['lr_decaying_rate'])
        hard_update(self.actor_target, self.actor)

        self.critic = Critic(input_size, output_size, configs)
        self.critic_target.to(self.device)
        self.critic_target = Critic(input_size, output_size, configs)
        self.critic_target.to(self.device)
        self.critic_optim = optim.Adam(
            self.critic.parameters(), configs['critic']['lr'])
        self.critic_lr_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optim, step_size=configs['critic']['lr_decaying_epoch'], gamma=configs['critic']['lr_decaying_rate'])
        hard_update(self.critic_target, self.critic)

        self.experience_replay = ReplayMemory(
            configs['experience_replay_size'])
        self.action_noise = OrnsteinUhlenbeckProcess(
            size=output_size, theta=configs['ou']['theta'], mu=configs['ou']['mu'], sigma=configs['ou']['sigma'])
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        self.actor.eval()
        mu = self.actor(state.float())
        self.actor.train()
        mu = mu.data

        if self.action_noise is not None:
            noise = torch.Tensor(self.action_noise.noise()).to(
                self.device)
            mu += noise

        mu = mu.clamp(1, -1)
        return mu

    def update(self, epoch):
        if len(self.experience_replay) <= self.configs['batch_size']:
            return
        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(
            batch.next_state).to(self.device)

        # get action and the state value from each target
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch.detach())

        # calc target
        reward_batch = reward_batch.unsqueeze(1)
        expected_values = reward_batch + self.gamma*next_state_action_values

        # critic network update
        self.critic_optim.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = self.critierion(
            state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optim.step()

        # actor network update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # update target
        soft_update(self.actor_target, self.actor)
        soft_update(self.critic_target, self.critic)

        return value_loss.item(), policy_loss.item()

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(state, action, reward, next_state)

    def hyperparams_update(self):
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
