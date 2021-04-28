import torch
import torch.nn as nn
import torch.nn.functional as f
import traci
import os


class MainAgent():
    def __init__(self, file_path, time_data, device, configs):
        self.configs = configs
        if configs['network'] == "cross":
            from Agent.crossAgent import CrossAgent
            self.network = CrossAgent(file_path, time_data, device, configs)
        elif configs['network'] == "grid":
            from Agent.gridAgent import GridAgent
            self.network = GridAgent(file_path, time_data, device, configs)


class BaseAgent():
    def __init__(self, file_path, time_data, device, configs):
        self.file_path = file_path
        self.time_data = time_data
        self.action_size = configs['EXP_CONFIGS']['action_size']
        self.state_size = configs['EXP_CONFIGS']['state_space']
        self.max_steps=float(configs['EXP_CONFIGS']['max_steps'])

        from Agent.Algorithm.dqn import DQN
        from Agent.Algorithm.ddpg import DDPG

        self.dqn_model = DQN(
            self.state_size, configs['AGENT_CONFIGS']['dqn']['action_space'], device, configs['AGENT_CONFIGS']['dqn'])
        self.ddpg_model = DDPG(
            self.state_size+1, 1, device, configs['AGENT_CONFIGS']['ddpg'])

        self.dqn_loss = 0
        self.ddpg_value_loss = 0
        self.ddpg_policy_loss = 0

    def get_action(self, state, num_agent):
        raise NotImplementedError

    def update(self, epoch):
        raise NotImplementedError

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('dqn/epsilon', self.dqn_model.epsilon, epoch)
        writer.add_scalar(
            'dqn/lr', self.dqn_model.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar(
            'ddpg/actor_lr', self.ddpg_model.actor_optim.param_groups[0]['lr'], epoch)
        writer.add_scalar(
            'ddpg/critic_lr', self.ddpg_model.actor_optim.param_groups[0]['lr'], epoch)
        writer.add_scalar('dqn/loss',self.dqn_loss/self.max_steps,epoch)
        writer.add_scalar('ddpg/value_loss',self.ddpg_value_loss/self.max_steps,epoch)
        writer.add_scalar('ddpg/policy_loss',self.ddpg_policy_loss/self.max_steps,epoch)
        writer.add_scalar('ddpg/total_loss',(self.ddpg_value_loss+self.ddpg_policy_loss)/self.max_steps,epoch)
        self.dqn_loss = 0
        self.ddpg_value_loss = 0
        self.ddpg_policy_loss = 0

    def save_weight(self, epoch):
        if epoch % 50 == 0:
            torch.save(self.dqn_model.behaviorQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'behaviorDQN.pt'))
            torch.save(self.dqn_model.targetQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'targetDQN.pt'))

            torch.save(self.ddpg_model.actor.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'actor_DDPG.pt'))
            torch.save(self.ddpg_model.actor_target.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'actor_targetDDPG.pt'))
            torch.save(self.ddpg_model.critic.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'critic_DDPG.pt'))
            torch.save(self.ddpg_model.critic_target.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'critic_targetDDPG.pt'))

    def load_weight(self, time_data):
        self.dqn_model.behaviorQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'behaviorDQN.pt')))
        self.dqn_model.targetQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'targetDQN.pt')))

        self.ddpg_model.actor.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'actor_DDPG.pt')))
        self.ddpg_model.actor_target.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'actor_targetDDPG.pt')))
        self.ddpg_model.critic.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'critic_DDPG.pt')))
        self.ddpg_model.critic_target.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'critic_targetDDPG.pt')))
