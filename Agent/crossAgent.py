from Agent.baseAgent import BaseAgent
import torch

AGENT_CONFIGS = {
    'ddpg': {
        'actor': {'fc': [50, 50, 50], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.99},
        'critic': {'fc': [50, 50, 50], 'lr': 1e-5, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.99},
        'experience_replay_size': 1e4,
        'batch_size': 32,
        'ou': {'theta': 0.15, 'sigma': 0.2, 'mu': 0.0},
        'gamma': 0.999,
        'action_space': [-1.0, 1.0],
        'init_train_ddpg':3000,
        'gym_mode':False,
    },
    'dqn': {
        'fc': [100, 100],
        'epsilon': 0.5,
        'epsilon_decaying_rate': 0.99,
        'epsilon_final': 0.001,
        'experience_replay_size': 1e4,
        'batch_size': 32,
        'lr': 1e-4,
        'lr_decaying_epoch': 50,
        'lr_decaying_rate': 0.5,
        'gamma': 0.999,
        'action_space': 3,
        'update_type': 'soft',
        'tau':0.99,
        'target_update_period': 20,
        'gym_mode':False,
    },
}


class CrossAgent(BaseAgent):
    def __init__(self, file_path, time_data, device, configs):
        if configs['mode'] != 'load_train':
            configs['AGENT_CONFIGS'] = AGENT_CONFIGS
        super(CrossAgent, self).__init__(file_path, time_data, device, configs)

    def get_action(self, states, num_agent):
        actions = list()
        if num_agent != 0:
            for state in states:
                # direction
                dqn_action = self.dqn_model.get_action(state.view(1, -1))
                # accel
                ddpg_action = self.ddpg_model.get_action(
                    torch.cat((state.view(1, -1), dqn_action.detach().clone()), dim=1))
                actions.append(torch.cat((ddpg_action, dqn_action), dim=1))
        if len(actions) != 0:
            actions = torch.cat(actions, dim=0).detach().clone()
        return actions

    def update(self, epoch, num_agent):
        if num_agent == 0:
            return
        next_action, self.dqn_loss = self.dqn_model.update(epoch)
        self.ddpg_value_loss, self.ddpg_policy_loss = self.ddpg_model.update(
            next_action, epoch)

    def save_replay(self, state, action, reward, next_state, num_agent):
        if type(action) == list or num_agent == 0:
            return
        else:
            for s, a, r, n_s in zip(state, action, reward, next_state):
                s, a, r, n_s = s.view(-1, self.state_size), a.view(-1,
                                                                   self.action_size), r, n_s.view(-1, self.state_size)
                self.dqn_model.save_replay(
                    s, a, r, n_s)
                self.ddpg_model.save_replay(s, a, r, n_s)
            return

    def hyperparams_update(self):
        self.dqn_model.hyperparams_update()
        self.ddpg_model.hyperparams_update()
