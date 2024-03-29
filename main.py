import os
import sys
import argparse
import random
import numpy as np
from torch._C import device
import traci
import time
from configs import DEFAULT_CONFIGS
from sumolib import checkBinary
from torch.utils.tensorboard import SummaryWriter
from utils import *
import traci.constants as tc
import torch
# 인자를 가져오는 함수




def parse_args(args):
    parser = argparse.ArgumentParser()

    # 기본 옵션
    parser.add_argument('mode', type=str, choices=[
                        'train', 'simulate', 'test', 'load_train','gym'])
    if 'gym' in parser.parse_known_args(args)[0].mode.lower():
        parser.add_argument('--algorithm',type=str,default='ddpg',choices=['ddpg','dqn','ppo'])

    # 추가 옵션
    # choose road network
    parser.add_argument('--network', type=str, default='cross')
    # display the monitor
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    # replay_option (test,load_train)
    parser.add_argument('--time_data', type=str, default=None)


    parser.add_argument('--step_length', type=float, default=1)
    #parser.add_argument('--agent', type=str)
    # algorithm decision
    #parser.add_argument('--alg', type=str, default='algorithm')
    return parser.parse_known_args(args)[0]


def test(time_data, device, configs, sumoBinary, sumoConfig):
    STEP_LENGTH=configs['step_length']
    sumoCmd = [sumoBinary, "-c", sumoConfig,"--step-length",str(STEP_LENGTH)]
    file_path = os.path.dirname(os.path.abspath(__file__))
    # 알고리즘 평가
    from Env.baseEnv import Env
    from Agent.baseAgent import MainAgent

    agent = MainAgent(file_path, time_data, device, configs).network
    agent.load_weight(time_data)
    agent.eval()
    traci.start(sumoCmd)
    step = 0
    env = Env(file_path, device, configs)
    state, num_agent,done = env.init()
    total_reward = 0
    ##########
    gen_agent_num = len(env.gen_agent_list)
    speed_state = torch.zeros(
        (gen_agent_num, 3), dtype=torch.float, device=env.device)

    prev_lane = torch.zeros(
        (gen_agent_num, 1), dtype=torch.float, device=env.device)
    penalty = torch.zeros(
        (gen_agent_num, 1), dtype=torch.float, device=env.device)

    follower_state= torch.zeros(
        (gen_agent_num, 3), dtype=torch.float, device=env.device)
    ##########
    tik = time.time()
    with torch.no_grad():
        while step < configs['EXP_CONFIGS']['max_steps']:
            action = agent.get_action(state, num_agent,done)

            next_state, reward, done,num_agent = env.step(action, step)
            step += STEP_LENGTH
            # arrived_vehicles += 해주는 과정 필요
            #agent.save_replay(state, action, reward, next_state, num_agent)
            #agent.update(epoch, num_agent)
            ###########
            eval_set_avg_speed(env, speed_state)
            eval_set_num_lane_change(env, penalty, prev_lane)
            eval_set_follower_rel_speed(env, follower_state)
            # print(state, action)
            ###########
            state = next_state
            total_reward += reward.sum()
        
    traci.close()
    tok = time.time()
    
    ##########
    print('avg speed: ',eval_get_avg_speed(speed_state))
    print('lane change:', eval_get_num_lane_change(penalty))
    print('follower speed:', eval_get_follower_rel_speed(follower_state))
    ##########

    print("Time:{:.3f}, Reward: {:.3f}".format(tok-tik, total_reward))


def train(time_data, device, configs, sumoBinary, sumoConfig):
    # agent 체크
    STEP_LENGTH=configs['step_length']
    sumoCmd = [sumoBinary, "-c", sumoConfig,"--step-length",str(STEP_LENGTH)]
    # config 값 세팅하고, 지정된 알고리즘으로 트레이닝
    file_path = os.path.dirname(os.path.abspath(__file__))
    from Agent.baseAgent import MainAgent
    from Env.baseEnv import Env
    agent = MainAgent(file_path, time_data, device, configs).network
    # training data 경로 설정
    writer = SummaryWriter(os.path.join(file_path, 'training_data', time_data))
    # Config 세팅
    epoch = 0

    env = Env(file_path, device, configs)
    gen_agent_num = len(env.gen_agent_list)
    speed_state = torch.zeros((gen_agent_num, 3), dtype=torch.float, device=env.device)
    # saveparams여기 쯤 넣어주면 될듯, 하이퍼파라미터 저장
    if configs['mode'] == 'train':
        save_params(file_path, time_data, configs)
    else:
        agent.load_weight(file_path, time_data)
    best_reward=0.0

    for epoch in range(configs['EXP_CONFIGS']['start_epoch'], configs['EXP_CONFIGS']['epochs']+1):
        traci.start(sumoCmd)
        step = 0.0
        env = Env(file_path, device, configs)
        state, done, num_agent = env.init()
        total_reward = 0.0
        tik = time.time()
        act_list = list()

        while step < configs['EXP_CONFIGS']['max_steps']:
            action = agent.get_action(state, num_agent,done)
            next_state, reward,done, num_agent = env.step(action, step)
            step += STEP_LENGTH
            # arrived_vehicles += 해주는 과정 필요
            
            show_actions(writer, action, epoch, step,act_list)
            agent.save_replay(state, action, reward, next_state, done, num_agent)
            state = next_state
            # print(state)
            total_reward += reward.sum().data
            eval_set_avg_speed(env, speed_state)
            agent.update(epoch, num_agent)
        traci.close()
        print('='*30)
        tok = time.time()
        agent.hyperparams_update()
        agent.target_update(epoch)
        # Tensorboard 가져오기
        #show_actions(writer, action, num_agent, step,act_list)
        update_tensorBoard(writer, agent, env, epoch, configs,act_list)
        agent.save_weight(epoch,best_reward,total_reward)
        best_reward=max(best_reward,total_reward)
        print("Epoch: {:4d}, Time:{:.3f}, Reward: {:.3f}".format(epoch, tok-tik, total_reward))
        print('avg speed: ',eval_get_avg_speed(speed_state))
        print('='*30)

    writer.close()


def simulate(flags,device, configs, sumoBinary, sumoConfig):
    STEP_LENGTH=configs['step_length']
    sumoCmd = [sumoBinary, "-c", sumoConfig,"--step-length",str(STEP_LENGTH)]
    traci.start(sumoCmd)
    file_path = os.path.dirname(os.path.abspath(__file__))
    from Env.baseEnv import Env
    env=Env(file_path=file_path,device=device,configs=configs)
    env.init()
    traci.simulation.subscribe()
    step = 0.0
    ##########
    gen_agent_num = len(env.gen_agent_list)
    speed_state = torch.zeros(
        (gen_agent_num, 3), dtype=torch.float, device=env.device)

    prev_lane = torch.zeros(
        (gen_agent_num, 1), dtype=torch.float, device=env.device)
    penalty = torch.zeros(
        (gen_agent_num, 1), dtype=torch.float, device=env.device)

    follower_state= torch.zeros(
        (gen_agent_num, 3), dtype=torch.float, device=env.device)
    ##########
    while step < configs['EXP_CONFIGS']['max_steps']:
        
        env.add_agent(step)
        traci.simulationStep()  # agent.step안에 들어가야함
        env.agent_update()

        step += STEP_LENGTH
        ###########
        eval_set_avg_speed(env, speed_state)
        eval_set_num_lane_change(env, penalty, prev_lane)
        eval_set_follower_rel_speed(env, follower_state)
        ###########    
    traci.close()
    ##########
    print('avg speed: ',eval_get_avg_speed(speed_state))
    print('lane change:', eval_get_num_lane_change(penalty))
    print('follower speed:', eval_get_follower_rel_speed(follower_state))
    ##########


def main(args):
    flags = parse_args(args)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and flags.gpu else 'cpu')
    file_path = os.path.dirname(os.path.abspath(__file__))
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    # file name
    #seed
    random_seed = flags.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    gen_training_data_path = os.path.join(
        file_path, 'training_data')
    if os.path.exists(gen_training_data_path) == False:
        os.mkdir(gen_training_data_path)

    # enviornment 체크
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # gui 확인
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # config 파일에서 데이터 가져오기
    if flags.mode == 'test' or flags.mode == 'load_train':
        configs = load_params(file_path, flags.time_data)
        time_data = flags.time_data
        # configs['mode'] == 'train'
    else:  # simulate or train
        configs = DEFAULT_CONFIGS
        configs.update(vars(flags))
        configs['time_data'] = time_data
        #configs['agent'] = flags.agent.lower()
        # 어떤 네트워크인지 체크
        from Network.baseNetwork import mainNetwork
        network = mainNetwork(file_path, configs).network
        network.generate_cfg(True, configs['mode'])

    configs['EXP_CONFIGS']['start_epoch'] = flags.start_epoch  # load용
    configs['EXP_CONFIGS']['epochs'] = flags.epochs
    configs['step_length']=flags.step_length
    # 모드 결정 및 실행
    if flags.mode.lower() == 'train' or flags.mode.lower() == 'load_train':
        sumoConfig = os.path.join(
            file_path, 'Net_data', '{}.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        train(time_data, device, configs, sumoBinary, sumoConfig)
    elif flags.mode.lower() == 'test':
        configs['mode']='test'
        sumoConfig = os.path.join(  # time인지 file_name인지 명시
            file_path, 'Net_data', '{}.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        test(flags.time_data, device, configs, sumoBinary, sumoConfig)
    elif 'gym' == flags.mode.lower():
        from Agent.gymAgent import GymLearner
        learner=GymLearner(flags,device,configs)
        learner.run()
        return
    else:  # simulate
        sumoConfig = os.path.join(
            file_path, 'Net_data', '{}.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        simulate(flags, device,configs, sumoBinary, sumoConfig)


if __name__ == '__main__':
    main(sys.argv[1:])
