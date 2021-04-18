from Agent.baseAgent import BaseAgent
import torch

AGENT_CONFIGS = {
    'ddpg': {
        'actor': {'fc': [200, 300], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.99},
        'critic': {'fc': [200, 300], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.99},
        'experience_replay_size': 1e5,
        'batch_size': 32,
        'ou': {'theta': 0.15, 'sigma': 0.2, 'mu': 0.0},
        'gamma': 0.999,
    },
    'dqn': {
        'fc': [400, 300],
        'epsilon': 0.9,
        'epsilon_decaying_rate': 0.99,
        'epsilon_final': 0.1,
        'experience_replay_size': 1e5,
        'batch_size': 32,
        'lr': 1e-3,
        'lr_decaying_epoch': 50,
        'lr_decaying_rate': 0.5,
        'gamma': 0.999,
    },
}


class CrossAgent(BaseAgent):
    def __init__(self, file_path, configs):
        configs['AGENT_CONFIGS'] = AGENT_CONFIGS
        super(CrossAgent, self).__init__(file_path, configs)

    def get_action(self, states, num_agent):
        actions = list()
        if num_agent != 0:
            for state in states:
                dqn_action = self.dqn_model.get_action(state.view(1, -1))
                ddpg_action = self.ddpg_model.get_action(
                    torch.cat((state.view(1, -1), dqn_action.detach().clone()), dim=1))
                actions.append(torch.cat((dqn_action, ddpg_action), dim=1))
        if len(actions) != 0:
            actions = torch.cat(actions, dim=0).detach().clone()
        return actions

    def update(self, epoch):
        self.dqn_loss = self.dqn_model.update(epoch)
        self.ddpg_value_loss, self.ddpg_policy_loss = self.ddpg_model.update(
            epoch)

    def save_replay(self, state, action, reward, next_state, num_agent):
        if type(action) == list or num_agent == 0:
            return
        else:
            for s, a, r, n_s in zip(state, action, reward, next_state):
                self.dqn_model.save_replay(s, a, r, n_s)
                self.ddpg_model.save_replay(s, a, r, n_s)
            return

    def hyperparams_update(self):
        self.dqn_model.hyperparams_update()
        self.ddpg_model.hyperparams_update()
