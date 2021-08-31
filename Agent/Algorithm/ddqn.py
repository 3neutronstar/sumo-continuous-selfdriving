import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import random
from Agent.Algorithm.utils import ReplayMemory
from Agent.Algorithm.utils import hard_update, soft_update
from collections import namedtuple

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, configs):
        self.configs = configs
        super(QNetwork, self).__init__()
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


class DDQN():
    def __init__(self, input_size, output_size,mode, device, configs):
        self.configs = configs
        self.device = device
        self.behaviorQ = QNetwork(input_size, output_size, configs)
        self.behaviorQ.to(self.device)
        self.targetQ = QNetwork(input_size, output_size, configs)
        self.targetQ.to(self.device)
        self.mode=mode
        if self.mode=='train':
            hard_update(self.targetQ, self.behaviorQ)
        
        self.transition = namedtuple('Transition',
                                        ('state', 'action', 'reward', 'next_state'))
        self.experience_replay = ReplayMemory(
            configs['experience_replay_size'],self.transition)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.behaviorQ.parameters(), lr=configs['lr'],weight_decay=5e-4)
        self.epsilon = configs['epsilon']
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, configs['lr_decaying_epoch'], configs['lr_decaying_rate'])
        self.epsilon_decaying_rate = configs['epsilon_decaying_rate']
        self.final_epsilon = configs['epsilon_final']

        self.running_loss = 0

    def get_action(self, state):
        if self.mode=='train':
            if random.random() > self.epsilon :  # epsilon greedy
                with torch.no_grad():
                    q = self.behaviorQ(state)
                    action = torch.max(q, dim=1)[1]
            else:
                action = torch.tensor([random.randint(0, self.configs['action_space']-1)], device=self.device)
        else: # 학습이 아닐때
            with torch.no_grad():
                q = self.targetQ(state)
                action = torch.max(q, dim=1)[1]
        return action.view(1, 1)

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(
            state, action, reward, next_state)  # 0 index인 이유는 ddpg와 섞이기 때문

    def update(self, epoch):
        if len(self.experience_replay) < self.configs['batch_size']*10:
            return 0
        self.targetQ.eval()
        self.behaviorQ.train()

        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = self.transition(*zip(*transitions))
        # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], dim=0).to(self.device)
        state_batch = torch.cat(batch.state, dim=0).to(self.device)
        action_batch = torch.cat(batch.action, dim=0).to(self.device)

        # reward_batch = torch.cat(torch.tensor(batch.reward, dim=0)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 칼럼을 선택
        state_action_values = self.behaviorQ(
            state_batch).gather(1, action_batch.view(-1, 1).to(torch.int64))  # for 3D

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        next_state_values = torch.zeros(
            self.configs['batch_size'], device=self.device, dtype=torch.float)
            
            
        #next_state_values[non_final_mask] = self.targetQ(non_final_next_states).detach().max(1)[0]#DQN #target의 q에서 맥스를 사용
        behavior_action=torch.argmax(self.behaviorQ(non_final_next_states),dim=1,keepdim=True)
        next_state_values[non_final_mask] = self.targetQ(non_final_next_states).detach().gather(dim=1,index=behavior_action).view(-1)#DDQN # behavior Q에서 max action을 사용하여 target Q value 가짐
        
        # 기대 Q 값 계산
        expected_state_action_values = (
            next_state_values * self.configs['gamma']) + reward_batch

        # loss 계산
        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))


        self.running_loss += loss.item()

        self.optimizer.zero_grad()
        # 모델 최적화
        loss.backward()
        for param in self.behaviorQ.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().clone().item()

    def hyperparams_update(self):
        self.lr_scheduler.step()
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decaying_rate
            
    def target_update(self,epoch):
        # target update
        if self.configs['update_type'] == 'soft':
            soft_update(self.targetQ, self.behaviorQ, self.configs['tau'])
        else:
            if epoch % self.configs['target_update_period'] == 0:
                hard_update(self.targetQ, self.behaviorQ)
    
    def eval(self):
        self.behaviorQ.eval()
        self.targetQ.eval()