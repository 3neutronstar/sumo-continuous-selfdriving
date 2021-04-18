import os
import sys
import argparse
import traci
import time
from configs import DEFAULT_CONFIGS
from sumolib import checkBinary
from torch.utils.tensorboard import SummaryWriter
from utils import update_tensorBoard, save_params, load_params
import traci.constants as tc
import torch
# 인자를 가져오는 함수

#


def parse_args(args):
    parser = argparse.ArgumentParser()

    # 기본 옵션
    parser.add_argument('mode', type=str, choices=[
                        'train', 'simulate', 'test', 'load_train'])

    # 추가 옵션
    # choose road network
    parser.add_argument('--network', type=str, default='cross')
    # display the monitor
    parser.add_argument('--disp', type=bool, default=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    # replay_option (test,load_train)
    parser.add_argument('--time_data', type=str, default=None)
    #parser.add_argument('--agent', type=str)
    # algorithm decision
    #parser.add_argument('--alg', type=str, default='algorithm')
    return parser.parse_known_args(args)[0]


def test(time_data, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    file_path = os.path.dirname(os.path.abspath(__file__))
    # 알고리즘 평가
    from Env.baseEnv import Env
    from Agent.baseAgent import MainAgent

    agent = MainAgent(file_path, time_data, configs).network
    # load_weight(flags.replay_name)
    # load_params()
    for epoch in range(configs['EXP_CONFIGS']['start_epoch'], configs['EXP_CONFIGS']['epochs']):
        traci.start(sumoCmd)
        step = 0
        env = Env(configs)
        state, num_agent = env.init()
        total_reward = 0
        tik = time.time()
        while step < configs['EXP_CONFIGS']['max_steps']:
            action = agent.get_action(state, num_agent)
            next_state, reward, num_agent = env.step(action, step)
            step += 1
            # arrived_vehicles += 해주는 과정 필요
            agent.save_replay(state, action, reward, next_state, num_agent)
            agent.update(epoch, num_agent)
            state = next_state
            total_reward += reward.sum()

        traci.close()
        #tok = time.time()
        agent.hyperparams_update()
        agent.save_weight(epoch)
        epoch += 1
        #print("Time:{}, Reward: {}".format(tok-tik, total_reward))
    


def train(time_data, configs, sumoBinary, sumoConfig):
    # agent 체크
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    # config 값 세팅하고, 지정된 알고리즘으로 트레이닝
    file_path = os.path.dirname(os.path.abspath(__file__))
    from Agent.baseAgent import MainAgent
    from Env.baseEnv import Env
    agent = MainAgent(file_path, time_data, configs).network
    # training data 경로 설정
    writer = SummaryWriter(os.path.join(file_path, 'training_data', time_data))
    # Config 세팅
    epoch = 0

    # saveparams여기 쯤 넣어주면 될듯, 하이퍼파라미터 저장
    if configs['mode'] == 'train':
        save_params(file_path, time_data, configs)
    else:
        agent.load_weight(file_path, time_data)

    for epoch in range(configs['EXP_CONFIGS']['start_epoch'], configs['EXP_CONFIGS']['epochs']):
        traci.start(sumoCmd)
        step = 0
        env = Env(configs)
        state, num_agent = env.init()
        total_reward = 0
        tik = time.time()
        while step < configs['EXP_CONFIGS']['max_steps']:
            action = agent.get_action(state, num_agent)
            next_state, reward, num_agent = env.step(action, step)
            step += 1
            # arrived_vehicles += 해주는 과정 필요
            agent.save_replay(state, action, reward, next_state, num_agent)
            agent.update(epoch, num_agent)
            state = next_state
            total_reward += reward.sum()

        traci.close()
        tok = time.time()
        agent.hyperparams_update()
        # Tensorboard 가져오기
        update_tensorBoard(writer, agent, env, epoch, configs)
        agent.save_weight(epoch)
        epoch += 1
        print("Time:{}, Reward: {}".format(tok-tik, total_reward))
    writer.close()


def simulate(flags, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    traci.start(sumoCmd)
    traci.simulation.subscribe()
    step = 0
    while step < configs['EXP_CONFIGS']['max_steps']:
        traci.simulationStep()  # agent.step안에 들어가야함
        step += 1

    traci.close()


def main(args):
    flags = parse_args(args)
    file_path = os.path.dirname(os.path.abspath(__file__))
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    # file name

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
        # Argument 호출
        configs['network'] = flags.network.lower()
        configs['mode'] = flags.mode.lower()
        configs['time_data'] = time_data
        #configs['agent'] = flags.agent.lower()
        # 어떤 네트워크인지 체크
        from Network.baseNetwork import mainNetwork
        network = mainNetwork(file_path, configs).network
        network.generate_cfg(True, configs['mode'])

    configs['EXP_CONFIGS']['start_epoch'] = flags.start_epoch  # load용
    configs['EXP_CONFIGS']['epochs'] = flags.epochs

    # 모드 결정 및 실행
    if flags.mode.lower() == 'train' or flags.mode.lower() == 'load_train':
        sumoConfig = os.path.join(
            file_path, 'Net_data', '{}.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        train(time_data, configs, sumoBinary, sumoConfig)

    elif flags.mode.lower() == 'test':
        sumoConfig = os.path.join(  # time인지 file_name인지 명시
            file_path, 'Net_data', '{}.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        test(flags, configs, sumoBinary, sumoConfig)

    else:  # simulate
        sumoConfig = os.path.join(
            file_path, 'Net_data', '{}.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        simulate(flags, configs, sumoBinary, sumoConfig)


if __name__ == '__main__':
    main(sys.argv[1:])
