from Network.cross import NET_CONFIGS
import os
import sys
import argparse
import traci
import time
from configs import DEFAULT_CONFIGS
from sumolib import checkBinary
from torch.utils.tensorboard import SummaryWriter
from utils import update_tensorBoard
import traci.constants as tc
from Env.baseEnv import Env
# 인자를 가져오는 함수

#
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
    # agent decision
    #parser.add_argument('--agent', type=str)
    # algorithm decision
    #parser.add_argument('--alg', type=str, default='algorithm')
    return parser.parse_known_args(args)[0]


def test(flags, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    # 알고리즘 평가
    traci.start(sumoCmd)
    step = 0
    while step < configs['EXP_CONFIGS']['max_step']:
        traci.simulationStep()
        step += 1


def train(flags, configs, sumoBinary, sumoConfig):
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    # config 값 세팅하고, 지정된 알고리즘으로 트레이닝
    file_path = os.path.dirname(os.path.abspath(__file__))
    #agent 체크
    from Agent.baseAgent import MainAgent

    agent = MainAgent(file_path,configs).network
    # training data 경로 설정
    writer = SummaryWriter(os.path.join(file_path, 'training_data'))
    #Config 세팅
    NUM_EPOCHS = configs['epochs']
    MAX_STEPS = configs['max_steps']
    epoch = 0

    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        step = 0
        env = Env(configs)
        state = env.init()
        total_reward = 0
        reward = 0

        while step < configs['EXP_CONFIGS']['max_step']:
            action = agent.get_action(state)
            next_state, reward = env.step(action)
            step += 1
            agent.save_replay(state, action, reward, next_state)
            agent.update(epoch)
            state = next_state
            total_reward += reward

        traci.close()
        epoch += 1
        agent.hyperparams_update()
        #Tensorboard 가져오기 
        update_tensorBoard(writer, agent, env, epoch)
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

    # config 파일에서 데이터 가져오기
    if flags.mode == 'test':
        configs = load_params(file_path)
    else:  # simulate or train
        configs = DEFAULT_CONFIGS

    # Argument 호출
    configs['network'] = flags.network.lower()
    configs['mode'] = flags.mode.lower()
    #configs['agent'] = flags.agent.lower()
    # 어떤 네트워크인지 체크
    from Network.baseNetwork import mainNetwork
    network = mainNetwork(file_path, configs).network
    network.generate_cfg(True, configs['mode'])
    # network.sumo_gui()



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
