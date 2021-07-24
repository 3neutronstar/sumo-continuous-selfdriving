import os
import json
import numpy as np
import torch
import traci
# 메인에서 끌어오는 형식으로 해보려고 한다.
def update_tensorBoard(writer, agent, env, epoch, configs,act_list):
    # agent.update_tensorBoard   Loss, Learning Rate, Epsilon dqn으로 설정해놓음
    agent.update_tensorboard(writer, epoch)
    # env.update_tensorBoard   Reward
    writer.add_scalar('episode/reward', env.reward.sum(),
                      epoch)
    actions=torch.cat(act_list,dim=0).T
    dqn_tensor = actions[1].float()
    ddpg_tensor = actions[0].float()

    writer.add_scalar('action/dqn_mean', torch.mean(dqn_tensor), epoch)
    writer.add_scalar('action/dqn_var', torch.var(dqn_tensor), epoch)
    writer.add_scalar('action/ddpg_mean', torch.mean(ddpg_tensor), epoch)
    writer.add_scalar('action/ddpg_var', torch.var(ddpg_tensor), epoch)
    env.reward = 0    
    #print(torch.var(dqn_tensor))
    #print(torch.var(ddpg_tensor))

def save_params(file_path, time_data, configs):
    with open(os.path.join(file_path, 'training_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)


def load_params(file_path, file_name):
    with open(os.path.join(file_path, 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
        return configs     


#rl 차량 평균속도
def eval_set_avg_speed(env, speed_state):
#speed_state는 ((총 agent 갯수)*(2)) 차원의 tensor이며, 열은 각각 'agent가 출발지부터 도착지까지 이동하는 데 걸린 time_step'과
#각 time_step에서 agent의 speed를 전부 합한 값을 의미     
    for i, agent in enumerate(env.gen_agent_list):
        if agent in env.agent_list:
            speed_state[i, 1] += 1 #elapsed time step
            speed_state[i, 2] += max(traci.vehicle.getSpeed(agent),0.0)

    return speed_state


def eval_get_avg_speed(speed_state):
    speed_state[:, 0] = torch.div(speed_state[:, 2], speed_state[:, 1]) #각 agent들의 매 time_step에서의 speed의 합을 총 time_step으로 나누고
    speed_state = speed_state[:, 0] #그 결과를 speed_state에 저장
    agent_avg_speed = speed_state.mean().item() 

    return agent_avg_speed


#차선 변경 횟수
def eval_set_num_lane_change(env, penalty, prev_lane):
    current_lane = torch.zeros(
        (len(env.gen_agent_list), 1), dtype=torch.float, device=env.device)

    for idx, agent in enumerate(env.gen_agent_list):
        if agent in env.agent_list:
            current_lane[idx]=traci.vehicle.getLaneIndex(agent)
    #print('prev_lane', prev_lane)
    #print('current_lane:', current_lane)

    pen=torch.eq(current_lane, prev_lane)
    #print('pen:', pen)
    for i in range (len(env.gen_agent_list)):
        if pen[i] == False:
            penalty[i] += 1.0
    #print('prev_lane_before_update:', prev_lane)

    for i in range (len(env.gen_agent_list)):
        prev_lane[i] = current_lane[i]

    #prev_lane = current_lane.clone()
    #print('prev_lane_after_update:', prev_lane)
    return penalty


def eval_get_num_lane_change(penalty):
    avg_lane_change = penalty.mean().item()
    return avg_lane_change


#follower 상대속도
#follower_speed는 ((follower 댓수) * (1)) 차원의 tensor이며
#traffic은  *3 차원의 tensor이며 모든 follower의 속도평균/모든 follower의 속도합/모든 follower 댓수를 의미
def eval_set_follower_rel_speed(env, follower_state):  
    for idx, agent in enumerate (env.gen_agent_list): #투입되는 모든 agent_list에서
        if agent in env.agent_list: #현재 존재하는 agent들에 대해 
            follower = traci.vehicle.getFollower(agent) #follower들을 구하고
            if follower[0] == '':
                continue
            else:
                follower_state[idx, 2] += traci.vehicle.getSpeed(follower[0]) #follwer의 현재 속도를 구하여 follwer_speed 텐서에 저장
                follower_state[idx, 1] += 1
                #결국 모든 agent_list들이 출발지에서 목적지로 향하는 동한 매 time step마다 발생되는 모든 follower들의 speed 총 합이
                #텐서형태로 저장되게 됨
    return follower_state

def eval_get_follower_rel_speed(follower_state):
    follower_state[:, 0] = torch.div(follower_state[:, 2], follower_state[:, 1])
    follower_state = follower_state[:, 0]

    follower_state[follower_state != follower_state] = 0 #remove NaN

    traffic = follower_state.mean().item()
    print('traffic:', traffic)
    return traffic

def show_actions(writer, action, epoch,step,act_list):    
    for a in action:
        act_list.append(a.view(1,-1))
