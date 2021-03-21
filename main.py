import os
import sys
import argparse
import traci
import configs
from sumolib import checkBinary

#인자를 가져오는 함수
def parse_args(args):
    parser = argparse.ArgumentParser()

    #기본 옵션 
    parser.add_argument('mode',type=str, choices=['train', 'simulate', 'test'])
    
    #추가 옵션
    parser.add_argument('--network', type=str)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default='algorithm')
    return parser.parse_known_args(args)[0]


def test(flags, configs, sumoConfig):    
    sumoCmd = [configs['gui'], "-c", sumoConfig]
    #알고리즘 평가

def train(flags, configs, sumoConfig):
    sumoCmd = [configs['gui'], "-c", sumoConfig]
    #config 값 세팅하고, 지정된 알고리즘으로 트레이닝 
    traci.start(sumoCmd)  
    step = 0
    while step < configs['max_step']:
        traci.simulationStep()
        step+=1

    traci.close()

def simulate(flags, configs, sumoConfig):
    sumoCmd = [configs['gui'], "-c", sumoConfig ]
    traci.start(sumoCmd)
    traci.simulation.subscribe()
    step = 0
    while step < configs['max_step']:
        traci.simulationStep()
        step += 1
        
    traci.close()




def main(args):
    flags = parse_args(args)

    #config 파일에서 데이터 가져오기
    configs = RL_configs.EXP_CONFIGS
    configs['mode'] = flags.mode
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    

    #네트워크 호출
    configs['network'] = "네트워크 이름" 

    # 어떤 네트워크인지 체크
    if configs['network'] == "네트워크 이름" :
        configs['NET_CONFIGS']["gird_num"] = 3
        #추가..
        # #net 생성


    # enviornment 체크
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    #gui 확인
    if flags.display == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    configs['gui'] = sumoBinary

    # 모드 결정 및 실행
    if configs['mode'] == 'train':
        sumoConfig = os.path.join(configs['current_path'], 'training_data', '_train.sumocfg') #중간 파일 경로 추가
        train(flags, configs, sumoConfig)
    
    elif configs['mode'] == 'test':
        sumoConfig = os.path.join(configs['current_path'], '_test.sumocfg')  #중간 파일 경로 추가
        test(flags, configs, sumoConfig)
   
    else:  # simulate
        sumoConfig = os.path.join(configs['current_path'], '_simulate.sumocfg')  #중간 파일 경로 추가
        simulate(flags, configs, sumoConfig)



if __name__ == '__main__':
    main(sys.argv[1:])
