import os
import json
import numpy as np
import torch

# 메인에서 끌어오는 형식으로 해보려고 한다.
def update_tensorBoard(writer, agent, env, epoch, configs,act_list):
    # agent.update_tensorBoard   Loss, Learning Rate, Epsilon dqn으로 설정해놓음
    agent.update_tensorboard(writer, epoch)
    # env.update_tensorBoard   Reward
    writer.add_scalar('episode/reward', env.reward.sum(),
                      epoch)
    i = 0
    dqn = list()
    ddpg = list()
    while i < len(act_list):
            #writer.add_histogram('action/dqn',epoch,act_list[i][1])
            #writer.add_histogram('action/ddpg', epoch, act_list[i][0])
            dqn.append(act_list[i][1])
            ddpg.append(act_list[i][0])
            i += 1
<<<<<<< HEAD
    dqn_tensor = torch.tensor(dqn)        
    ddpg_tensor = torch.tensor(ddpg)
=======
    writer.flush()
    env.reward = 0
>>>>>>> 20622a9aaa291b2e1e9e50cf47841c4259c04015

    writer.add_scalar('action/dqn_mean', torch.mean(dqn_tensor), epoch)
    writer.add_scalar('action/dqn_var', torch.var(dqn_tensor), epoch)
    writer.add_scalar('action/ddpg_mean', torch.mean(ddpg_tensor), epoch)
    writer.add_scalar('action/ddpg_var', torch.var(ddpg_tensor), epoch)
    env.reward = 0    

def save_params(file_path, time_data, configs):
    with open(os.path.join(file_path, 'training_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)


def load_params(file_path, file_name):
    with open(os.path.join(file_path, 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs

<<<<<<< HEAD
def show_actions(writer, action, epoch,step,act_list):    
    i = 0
    while i < len(action):
        act_list.append(action[i])
        i += 1
=======
def show_actions(writer, action, epoch,step,act_list):
    # if len(action) :
    #     while num_agent > 0:
    #         a = np.array(action)
    #         writer.add_scalar('step/direction', a[num_agent-1][1],step)
    #         writer.add_scalar('step/accel', a[num_agent-1][0],step)
    #         num_agent -1
    #         if step == 3600:
    #             break
    # if len(action) :
    #     i = 0
    #     while i < len(action):
    #         if action[i][1] == 0:
    #             act_list[0] += 1
    #         elif action[i][1] == 1:
    #             act_list[1] += 1
    #         else:
    #             act_list[2] += 1
    #         i += 1 
    #         if step == 3600:
    #             break
    # if step ==3600:
    #     writer.add_scalar('action/0',act_list[0], epoch)
    #     writer.add_scalar('action/1',act_list[1], epoch )
    #     writer.add_scalar('action/else',act_list[2], epoch )
    # if len(action):
    #     act = torch.Tensor(action)
    #     writer.add_histogram('actions', act, step)
    # i = 0
    # while i < len(action):
    #     act_list.append(action[i])
    #     i += 1

    for a in action:
        act_list.append(a)
>>>>>>> 20622a9aaa291b2e1e9e50cf47841c4259c04015
