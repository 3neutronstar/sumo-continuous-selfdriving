import os
import sys
import argparse
import traci
import time
from configs import EXP_CONFIGS
from sumolib import checkBinary

# 인자를 가져오는 함수


def parse_args(args):
    parser = argparse.ArgumentParser()

    # 기본 옵션
    parser.add_argument('mode', type=str, choices=[
                        'train', 'simulate', 'test'])

    # 추가 옵션

    # choose road network
    parser.add_argument('--net', type=str)
    # display the monitor
    parser.add_argument('--disp', type=bool, default=False)
    # algorithm decision
    parser.add_argument('--alg', type=str, default='algorithm')
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
    while step < configs['max_step']:
        traci.simulationStep()  # agent.step안에 들어가야함
        step += 1

    traci.close()


def simulate(flags, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    traci.start(sumoCmd)
    traci.simulation.subscribe()
    step = 0
    while step < configs['max_step']:
        traci.simulationStep()  # agent.step안에 들어가야함
        step += 1

    traci.close()


def main(args):
    flags = parse_args(args)
    running_path = os.path.dirname(os.path.abspath(__file__))

    # config 파일에서 데이터 가져오기
    if flags.mode == 'test':
        configs = load_params(running_path)
    else:  # simulate or train
        configs = EXP_CONFIGS

    # 네트워크 호출
    configs['network'] = flags.network.lower()

    # 어떤 네트워크인지 체크
    if configs['network'] == "네트워크 이름을 적어주세요":
        configs['NET_CONFIGS']["grid_num"] = 3
        # 추가..
        # #net 생성

    # enviornment 체크
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # gui 확인
    if flags.display == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    # 모드 결정 및 실행
    if flags.mode.lower() == 'train':
        sumoConfig = os.path.join(
            running_path, 'training_data', '_train.sumocfg')  # 중간 파일 경로 추가
        train(flags, configs, sumoBinary, sumoConfig)

    elif flags.mode.lower() == 'test':
        sumoConfig = os.path.join(
            running_path, '_test.sumocfg')  # 중간 파일 경로 추가
        test(flags, configs, sumoBinary, sumoConfig)

    else:  # simulate
        sumoConfig = os.path.join(
            running_path, '_simulate.sumocfg')  # 중간 파일 경로 추가
        simulate(flags, configs, sumoBinary, sumoConfig)


if __name__ == '__main__':
    main(sys.argv[1:])
