import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import random
from Agent.Algorithm.utils import ReplayMemory, Transition
from Agent.Algorithm.utils import hard_update, soft_update


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, configs):
        self.configs = configs
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = self._make_layers()
        print(self)

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


class DQN():
    def __init__(self, input_size, output_size, configs):
        self.configs = configs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.behaviorQ = QNetwork(input_size, output_size, configs)
        self.behaviorQ.to(self.device)
        self.targetQ = QNetwork(input_size, output_size, configs)
        self.targetQ.to(self.device)
        hard_update(self.targetQ, self.behaviorQ)
        self.experience_replay = ReplayMemory(
            configs['experience_replay_size'])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.behaviorQ.parameters(), lr=configs['lr'])
        self.epsilon = configs['epsilon']
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, configs['lr_decaying_epoch'], configs['lr_decaying_rate'])
        self.epsilon_decaying_rate = configs['epsilon_decaying_rate']
        self.final_epsilon = configs['epsilon_final']

    def get_action(self, state):
        if random.random() > self.epsilon:  # epsilon greedy
            with torch.no_grad():
                q = self.behaviorQ(state)
                action = torch.max(q, dim=1)[1]
        else:
            action = torch.tensor([random.randint(-1, 1)], device=self.device)
        return action.view(1, 1)

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(
            state, action, reward, next_state)

    def update(self, epoch):
        if len(self.experience_replay) < self.configs['batch_size']:
            return 0

        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = Transition(*zip(*transitions))

        # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], dim=0)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action, dim=0)  # 안쓰지만 배치함

        # reward_batch = torch.cat(torch.tensor(batch.reward, dim=0)
        reward_batch = torch.tensor(batch.reward)
        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 칼럼을 선택

        state_action_values = self.behaviorQ(
            state_batch).gather(1, action_batch)  # for 3D
        # state_action_values = self.behaviorQ(
        #     state_batch)
        # .max(1)[0].clone().float().unsqueeze(1)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        next_state_values = torch.zeros(
            self.configs['batch_size'], device=self.device, dtype=torch.float)

        next_state_values[non_final_mask] = self.targetQNetwork(
            non_final_next_states).max(1)[0].detach()  # .to(self.device)  # 자신의 Q value 중에서max인 value를 불러옴

        # 기대 Q 값 계산
        expected_state_action_values = (
            next_state_values * self.configs['gamma']) + reward_batch

        # loss 계산
        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))
        self.running_loss += loss
        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.behaviorQ.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # target update
        if self.configs['update_type'] == 'soft':
            soft_update(self.targetQ, self.behaviorQ, self.configs['tau'])
        else:
            if epoch % self.configs['target_update_period'] == 0:
                hard_update(self.targetQ, self.behaviorQ)
        return loss.detach().clone()

    def hyperparams_update(self):
        self.lr_scheduler.step()
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decaying_rate
