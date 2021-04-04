from Agent.baseAgent import BaseAgent
import torch

AGENT_CONFIGS = {
    'ddqg': {
        'actor': {'fc': [200, 300], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, },
        'critic': {'fc': [200, 300], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, },
        'experience_replay_size': 1e5,
        'batch_size': 32,
        'ou': {'theta': 0.15, 'sigma': 0.2, 'mu': 0.0},
        'gamma': 0.999,
    },
    'dqn': {
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
        super(BaseAgent, self).__init__(file_path, configs)

    def get_action(self, states):
        actions = list()

        for state in states:
            dqn_action = self.dqn_model(state)
            ddpg_action = self.ddpg_model(
                torch.cat((state, dqn_action.detach().clone()), dim=1))
            actions.append(torch.cat((dqn_action, ddpg_action), dim=1))
        actions = torch.cat(actions.detach().clone(), device=0)
        return actions

    def update(self, epoch):
        dqn_loss = self.dqn_model.update(epoch)
        ddpg_loss = self.ddpg_model.update(epoch)

    def save_replay(self, state, action, reward, next_state):
        self.dqn_model.save_replay(state, action, reward, next_state)
        self.ddpg_model.save_replay(state, action, reward, next_state)
