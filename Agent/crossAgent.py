from Agent.Algorithm.ppo import PPO
from Agent.Algorithm.ddqn import DDQN
from Agent.baseAgent import BaseAgent
import torch
import os
from Agent.Algorithm.dqn import DQN
from Agent.Algorithm.ddpg import DDPG

AGENT_CONFIGS = {
    'ddpg': {
        'actor': {'fc': [50, 50, 50], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.02},
        'critic': {'fc': [50, 50, 50], 'lr': 1e-4, 'lr_decaying_epoch': 50, 'lr_decaying_rate': 0.8, 'tau': 0.02},
        'experience_replay_size': 1e4,
        'batch_size': 32,
        'ou': {'theta': 0.15, 'sigma': 0.2, 'mu': 0.0},
        'gamma': 0.999,
        'action_space': [-1.0, 1.0],
        'init_train_ddpg':3000,
        'gym_mode':False,
        'initial_noise_scale':1.0,
        'final_noise_scale':0.01,
        'noise_reduce_rate':0.99,
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
        'update_type': 'hard',
        'tau':0.05,
        'target_update_period': 20,
        'gym_mode':False,
    },
}
class DDPGAgent(BaseAgent):
    def __init__(self, file_path, time_data, device, configs):
        super().__init__(file_path, time_data, device, configs)
        self.dqn_model = DQN(
            self.state_size, configs['AGENT_CONFIGS']['dqn']['action_space'],self.mode, device, configs['AGENT_CONFIGS']['dqn'])
        self.ddpg_model = DDPG(
            self.state_size+1, 1, self.mode,device, configs['AGENT_CONFIGS']['ddpg'])

        self.dqn_loss = 0.0
        self.ddpg_value_loss = 0.0
        self.ddpg_policy_loss = 0.0

    def get_action(self, states, num_agent,done):
        actions = list()
        if num_agent != 0:
            for i,state in enumerate(states):
                if done[i]==True:
                    continue
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
        next_action, dqn_loss = self.dqn_model.update(epoch)
        self.dqn_loss+=dqn_loss
        ddpg_value_loss, ddpg_policy_loss = self.ddpg_model.update(
            next_action, epoch)
        self.ddpg_value_loss+=ddpg_value_loss
        self.ddpg_policy_loss+=ddpg_policy_loss

    def save_replay(self, state, action, reward, next_state,done, num_agent):
        if type(action) == list or num_agent == 0:
            return
        else:
            for s, a, r, n_s,d in zip(state, action, reward, next_state,done):
                if d:
                    continue
                s, a, r, n_s = s.view(-1, self.state_size), a.view(-1,
                                                                   self.action_size), r, n_s.view(-1, self.state_size)
                self.dqn_model.save_replay(
                    s, a, r, n_s)
                self.ddpg_model.save_replay(s, a, r, n_s)
            return

    def hyperparams_update(self):
        self.dqn_model.hyperparams_update()
        self.ddpg_model.hyperparams_update()
    
    def target_update(self,epoch):
        self.dqn_model.target_update(epoch)

    def save_weight(self, epoch,best_reward,total_reward):
        if epoch!=0 and best_reward<total_reward:
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
            print("Model Save")

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
        writer.add_scalar('ddpg/noise_scale',self.ddpg_model.noise_scale,epoch)
        print("DQN LOSS:{:.4e} DDPG VALUE LOSS:{:.4e} POLICY LOSS:{:.4e}".format(self.dqn_loss/self.max_steps,self.ddpg_value_loss/self.max_steps,self.ddpg_policy_loss/self.max_steps))
        self.dqn_loss = 0.0
        self.ddpg_value_loss = 0.0
        self.ddpg_policy_loss = 0.0
    
    def eval(self):
        self.ddpg_model.eval()
        self.dqn_model.eval()

DDQN_AGENT_CONFIGS = {
    'ddqn1': { # direction
        'fc': [50, 50],
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
        'update_type': 'hard',
        'tau':0.05,
        'target_update_period': 20,
        'gym_mode':False,
    },
    'ddqn2': { # accel
        'fc': [50, 50],
        'epsilon': 0.5,
        'epsilon_decaying_rate': 0.99,
        'epsilon_final': 0.001,
        'experience_replay_size': 1e4,
        'batch_size': 32,
        'lr': 1e-4,
        'lr_decaying_epoch': 50,
        'lr_decaying_rate': 0.5,
        'gamma': 0.999,
        'action_space': 9,
        'update_type': 'hard',
        'tau':0.05,
        'target_update_period': 20,
        'gym_mode':False,
    },
}



class DDQNAgent(BaseAgent):
    def __init__(self, file_path, time_data, device, configs):
        super(DDQNAgent, self).__init__(file_path, time_data, device, configs)
        self.ddqn1 = DDQN(
            self.state_size, configs['AGENT_CONFIGS']['ddqn1']['action_space'],self.mode, device, configs['AGENT_CONFIGS']['ddqn1'])
        self.ddqn2 = DDQN(
            self.state_size, configs['AGENT_CONFIGS']['ddqn2']['action_space'], self.mode,device, configs['AGENT_CONFIGS']['ddqn2'])

        self.ddqn1_loss = 0.0
        self.ddqn2_loss = 0.0

    def get_action(self, states, num_agent,done):
        actions = list()
        if num_agent != 0:
            for i,state in enumerate(states):
                if done[i]==True:
                    continue
                # direction
                ddqn_action1 = self.ddqn1.get_action(state.view(1, -1))
                # accel
                ddqn_action2 = self.ddqn2.get_action(state.view(1,-1))
                actions.append(torch.cat((ddqn_action2, ddqn_action1), dim=1))
        if len(actions) != 0:
            actions = torch.cat(actions, dim=0).detach().clone()
        return actions

    def update(self, epoch, num_agent):
        if num_agent == 0:
            return
        self.ddqn1_loss+= self.ddqn1.update(epoch)
        self.ddqn2_loss+= self.ddqn2.update(epoch)

    def save_replay(self, state, action, reward, next_state, done,num_agent):

        if type(action) == list or num_agent == 0:
            return
        else:
            for s, a, r, n_s,d in zip(state, action, reward, next_state,done):
                if d :
                    continue
                s, a, r, n_s = s.view(-1, self.state_size), a.view(-1,
                                                                   self.action_size), r, n_s.view(-1, self.state_size)
                self.ddqn1.save_replay(
                    s, a[:,1], r, n_s)
                self.ddqn2.save_replay(s, a[:,0], r, n_s)
            return

    def hyperparams_update(self):
        self.ddqn1.hyperparams_update()
        self.ddqn2.hyperparams_update()

    def save_weight(self, epoch,best_reward,total_reward):
        self.ddqn1.optimizer.zero_grad()
        self.ddqn2.optimizer.zero_grad()
        if epoch!=0 and best_reward<total_reward:
            torch.save(self.ddqn1.behaviorQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'behaviorDDQN.pt'))
            torch.save(self.ddqn1.targetQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'targetDDQN.pt'))

            torch.save(self.ddqn2.behaviorQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'behaviorDDQN2.pt'))
            torch.save(self.ddqn2.targetQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'targetDDQN2.pt'))
            print("Model Save")

    def load_weight(self, time_data):
        self.ddqn1.behaviorQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'behaviorDDQN.pt')))
        self.ddqn1.targetQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'targetDDQN.pt')))
        self.ddqn2.behaviorQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'behaviorDDQN2.pt')))
        self.ddqn2.targetQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'targetDDQN2.pt')))
        print("load weight complete")

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('ddqn1/epsilon', self.ddqn1.epsilon, epoch)
        writer.add_scalar('ddqn2/epsilon', self.ddqn2.epsilon, epoch)
        writer.add_scalar(
            'ddqn1/lr', self.ddqn1.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('ddqn1/loss',self.ddqn1_loss/self.max_steps,epoch)
        writer.add_scalar(
            'ddqn2/lr', self.ddqn2.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('ddqn2/loss',self.ddqn2_loss/self.max_steps,epoch)
        print("DDQN1 LOSS:{:.4e} DDQN2 LOSS:{:.4e}".format(self.ddqn1_loss/self.max_steps,self.ddqn2_loss/self.max_steps))
        self.ddqn1_loss = 0.0
        self.ddqn2_loss = 0.0
        
    def target_update(self,epoch):
        self.ddqn1.target_update(epoch)
        self.ddqn2.target_update(epoch)

    def eval(self):
        self.ddqn1.eval()
        self.ddqn2.eval()



PPO_DDQN_AGENT_CONFIGS = {
    'ddqn1': { # direction
        'fc': [100, 100],
        'epsilon': 0.5,
        'epsilon_decaying_rate': 0.99,
        'epsilon_final': 0.001,
        'experience_replay_size': 1e5,
        'batch_size': 32,
        'lr': 1e-2,
        'lr_decaying_epoch': 50,
        'lr_decaying_rate': 0.5,
        'gamma': 0.99,
        'action_space': 3,
        'update_type': 'hard',
        'tau':0.05,
        'target_update_period': 100,
        'gym_mode':False,
    },
    'ppo': { # accel
        'actor':{'lr': 3e-4,
        'lr_decaying_epoch': 50,
        'lr_decaying_rate': 0.5,},
        'critic':{'lr': 1e-3,
        'lr_decaying_epoch': 50,
        'lr_decaying_rate': 0.5,},
        
        'gamma': 0.99,
        'eps_clips':0.2,
        'k_epochs':10,
        'action_space': 1,
        'gym_mode':False,
    },
}



class PPO_DDQN_Agent(BaseAgent):
    def __init__(self, file_path, time_data, device, configs):
        super(PPO_DDQN_Agent, self).__init__(file_path, time_data, device, configs)
        self.ddqn1 = DDQN(
            self.state_size, configs['AGENT_CONFIGS']['ddqn1']['action_space'],self.mode, device, configs['AGENT_CONFIGS']['ddqn1'])
        self.ppo = PPO(
            self.state_size, 
            configs['AGENT_CONFIGS']['ppo']['action_space'],
            K_epochs=configs['AGENT_CONFIGS']['ppo']['k_epochs'],
            eps_clip=configs['AGENT_CONFIGS']['ppo']['eps_clips'], 
            device=device, 
            configs=configs['AGENT_CONFIGS']['ppo'])
        self.device=device
        self.ddqn1_loss = 0.0
        self.ppo_loss = 0.0

    def get_action(self, states, num_agent,done):
        actions = list()
        if num_agent!=0:
            self.ddqn1.eval()
            self.ppo.policy_old.eval()
            for i,state in enumerate(states):
                if done[i]:
                    continue
                # direction
                ddqn_action1 = self.ddqn1.get_action(state.view(1, -1))
                # accel
                ppo_action = torch.from_numpy(self.ppo.select_action(state.view(1,-1))).to(self.device).view(1,-1)
                # print(ppo_action)
                actions.append(torch.cat((ppo_action, ddqn_action1), dim=1))
        if len(actions) != 0:
            actions = torch.cat(actions, dim=0).detach().clone()
        return actions

    def update(self, epoch, num_agent):
        if num_agent == 0:
            return
        self.ddqn1_loss+= self.ddqn1.update(epoch)
        self.ppo_loss+=self.ppo.update()

    def save_replay(self, state, action, reward, next_state, done,num_agent):
        if type(action) == list or num_agent==0:
            return
        else:
            for s, a, r, n_s,d in zip(state, action, reward, next_state,done):
                s, a, r, n_s,d = s.view(-1, self.state_size), a.view(-1,
                                                                self.action_size), r, n_s.view(-1, self.state_size),d.view(-1,1)
                if d:
                    self.ddqn1.save_replay(
                        s, a[:,1], r, None)
                else:
                    self.ddqn1.save_replay(
                        s, a[:,1], r, n_s)
                self.ppo.buffer.rewards.append(r)
                self.ppo.buffer.is_terminals.append(d)
            return

    def hyperparams_update(self):
        self.ddqn1.hyperparams_update()
        self.ppo.hyperparams_update()
        self.ppo.buffer.states=[]
        self.ppo.buffer.rewards=[]
        self.ppo.buffer.actions=[]
        self.ppo.buffer.logprobs=[]
        self.ppo.buffer.is_terminals=[]

    def save_weight(self, epoch,best_reward,total_reward):
        self.ddqn1.optimizer.zero_grad()
        self.ppo.optimizer.zero_grad()
        if epoch!=0 and best_reward<total_reward:
            torch.save(self.ddqn1.behaviorQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'behaviorDDQN.pt'))
            torch.save(self.ddqn1.targetQ.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'targetDDQN.pt'))

            torch.save(self.ppo.policy.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'ppo_policy.pt'))
            torch.save(self.ppo.policy_old.state_dict(), os.path.join(
                self.file_path, 'training_data', self.time_data, 'ppo_old_policy.pt'))
            print("Model Save")

    def load_weight(self, time_data):
        self.ddqn1.behaviorQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'behaviorDDQN.pt')))
        self.ddqn1.targetQ.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'targetDDQN.pt')))
        self.ppo.policy.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'ppo_policy.pt')))
        self.ppo.policy_old.load_state_dict(torch.load(
            os.path.join(self.file_path, 'training_data', time_data, 'ppo_old_policy.pt')))
        print("load weight complete")

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('ddqn1/epsilon', self.ddqn1.epsilon, epoch)
        writer.add_scalar(
            'ddqn1/lr', self.ddqn1.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('ddqn1/loss',self.ddqn1_loss/self.max_steps,epoch)
        writer.add_scalar(
            'ppo/lr', self.ppo.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('ppo/loss',self.ppo_loss/self.max_steps,epoch)
        print("ddqn1 LOSS:{:.4e} ppo LOSS:{:.4e}".format(self.ddqn1_loss/self.max_steps,self.ppo_loss/self.max_steps))
        self.ddqn1_loss = 0.0
        self.ppo_loss = 0.0
        
    def target_update(self,epoch):
        self.ddqn1.target_update(epoch)

    def eval(self):
        self.ddqn1.eval()
        self.ppo.policy_old.eval()



# class CrossAgent(DDQNAgent):
#     def __init__(self, file_path, time_data, device, configs):
#         if configs['mode'] != 'load_train':
#             configs['AGENT_CONFIGS'] = DDQN_AGENT_CONFIGS
#         super(CrossAgent, self).__init__(file_path, time_data, device, configs)
# class CrossAgent(DDPGAgent):
#     def __init__(self, file_path, time_data, device, configs):
#         if configs['mode'] != 'load_train':
#             configs['AGENT_CONFIGS'] = AGENT_CONFIGS
#         super(CrossAgent, self).__init__(file_path, time_data, device, configs)

class CrossAgent(PPO_DDQN_Agent):
    def __init__(self, file_path, time_data, device, configs):
        if configs['mode'] != 'load_train':
            configs['AGENT_CONFIGS'] = PPO_DDQN_AGENT_CONFIGS
        super(CrossAgent, self).__init__(file_path, time_data, device, configs)

