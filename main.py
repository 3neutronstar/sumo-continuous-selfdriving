from Network.cross import NET_CONFIGS
import os
import sys
import argparse
import traci
import time
from configs import DEFAULT_CONFIGS
from sumolib import checkBinary

# 인자를 가져오는 함수


def parse_args(args):
    parser = argparse.ArgumentParser()

    # 기본 옵션
    parser.add_argument('mode', type=str, choices=[
                        'train', 'simulate', 'test'])

    # 추가 옵션

    # choose road network
    parser.add_argument('--network', type=str)
    # display the monitor
    parser.add_argument('--disp', type=bool, default=False)
    # algorithm decision
    #parser.add_argument('--alg', type=str, default='algorithm')
    return parser.parse_known_args(args)[0]


def test(flags, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    # 알고리즘 평가
    traci.start(sumoCmd)
    step = 0
    while step < configs['max_step']:
        traci.simulationStep()
        step += 1


def train(flags, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    # config 값 세팅하고, 지정된 알고리즘으로 트레이닝
    traci.start(sumoCmd)
    step = 0
    # Agent
    agent = ALG(configs)
    # Env
    env = ENV(configs)
    # state init
    state = env.init()

    while step < configs['max_step']:
        agent.get_action(state)
        traci.simulationStep()  # agent.step안에 들어가야함
        next_state, reward = env.step(action)
        step += 1
        agent.update()

    traci.close()


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

    # config 파일에서 데이터 가져오기
    if flags.mode == 'test':
        configs = load_params(file_path)
    else:  # simulate or train
        configs = DEFAULT_CONFIGS

    # 네트워크 호출
    configs['network'] = flags.network.lower()
    configs['mode'] = flags.mode.lower()

    # 어떤 네트워크인지 체크
    from Network.baseNetwork import mainNetwork
    network = mainNetwork(file_path, configs).network
    configs['NET_CONFIGS'] = NET_CONFIGS
    network.generate_cfg(True, configs['mode'])
    # network.sumo_gui()
    # 네트워크의 congfig들을 받고, configs의 딕셔너리에 넣어주기 위함
    # NET부분에서 새로 생긴 configs들을 받아오는 작업이 필요하다.
    # run, 수모 돌리는 부분 필요

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

    # 모드 결정 및 실행
    if flags.mode.lower() == 'train':
        sumoConfig = os.path.join(
            file_path, 'training_data', '{}_train.sumocfg')  # 중간 파일 경로 추가
        train(flags, configs, sumoBinary, sumoConfig)

    elif flags.mode.lower() == 'test':
        sumoConfig = os.path.join(
            file_path, 'training_data', '{}_test.sumocfg')  # 중간 파일 경로 추가
        test(flags, configs, sumoBinary, sumoConfig)

    else:  # simulate
        sumoConfig = os.path.join(
            file_path, 'Net_data', '{}_simulate.sumocfg'.format(configs['network']))  # 중간 파일 경로 추가
        simulate(flags, configs, sumoBinary, sumoConfig)


if __name__ == '__main__':
    main(sys.argv[1:])
