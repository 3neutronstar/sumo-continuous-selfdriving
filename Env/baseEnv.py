'''
state size
1.changeLane 가능여부(좌/우)
2.앞차와의 거리
3.뒷차와의 거리
4.내 속도
5.내 레인 위치
6.내 edge상의 위치
7.내 차량이 갈 방향 (추후 추가 예정)
'''
'''
각 텐서 형태
state -> (agent갯수(1)) * (state갯수(8)) tensor
action -> (agent갯수(1) * (action_size(2)) tensor
reward -> (agent갯수(1)) * (1) tensor
'''
'''
done: 차량이 도착하거나 그 밖의 이유로 사라지는 경우
transition 입장에서 state와 action tensor는 존재하나 reward는 존재하지 않는 경우가 발생
reward가 없어지는 경우는 done mask를 이용해서 덮어씌워야 하는 기능이 필요할 가능성이 있음.
'''

import torch
import traci
import time
from copy import deepcopy
from configs import EXP_CONFIGS


ENV_CONFIGS = {
    'state_space': 8,
    'gen_agent_list': ['agent_0'],
    'action_size': 2 #방향(좌/우) / 속도(가속/감속)
}


class Env():
    #__init__에서 반영하는 변수는 추후 config.py로 이동할것
    def __init__(self, configs):
        self.env_configs = ENV_CONFIGS
        self.agent_list = list()
        self.gen_agent_list = self.env_configs['gen_agent_list']
        self.num_agent = 0
        self.state_space = self.env_configs['state_space']

        self.reward = torch.zeros((self.num_agent, 1), dtype=torch.float, device=self.device)

        self.observ_list = self.get_observ_list()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
  

    def init(self):
        state = self.collect_state()
        return state
    
    #agent 투입, 각 agent의 departure 간에 적절한 delay 삽입
    def add_agent(self, step):
        if step == 1:
            for id in self.gen_agent_list:
                traci.vehicle.add(vehID=id, routeID='route_0', typeID='car', departLane='random')


    #agent의 생성과 제거를 판단
    def agent_update(self):
        #도착한 agent 제거
        for idx, agent in enumerate(self.agent_list):
            if agent in traci.simulation.getArrviedIDList():
                self.agent_list.pop(idx) #agent_list에서 도착 agent 제거
                self.num_agent-=1
        
        #생성된 agent 추가
        for id in traci.simulation.getLoadedIDList():
            if traci.vehicle.getTypeID(id)=='rl_agent':
                self.agent_list.append(id)
                self.num_agent+=1
    

    def collect_state(self):         
        next_state = torch.zeros(
            (self.num_agent, self.state_space), dtype=torch.float, device=self.device)
        agent_state = torch.zeros(
            (1, self.state_space), dtype=torch.float, device=self.device) #agent_state는 agent별 state를 나타냄

        for i, agent in enumerate(self.agent_list):
            for idx,observ in enumerate(self.observ_list):
                agent_state[0, idx] = observ(agent)
                next_state[i,:] = next_state.clone().detach()

        return next_state


    def collect_reward(self):
        reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        agent_reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        #cum_reward = torch.like(reward) cumulative reward 필요한가?
        
        for idx, agent in enumerate(self.gen_agent_list):           
            agent_reward[idx] = traci.vehicle.getSpeed(agent)
            reward = agent_reward.clone().detach()
    
        return reward      
   
    def step(self, action, step):
        if self.num_agent == 0:
            pass
            #action mapping
        else:    
            for idx,agent in enumerate(self.agent_list):
                currentSpeed = traci.vehicle.getSpeed(agent)
                acc = action[idx, 0]
                traci.vehicle.setSpeed(agent, currentSpeed+acc)
                
                if action[idx, 1] == 1:
                    self.actionLeftLane(agent)
                elif action[idx, 1] == -1:
                    self.actionRightLane(agent)
                else:
                    self.actionStayLane(agent)        
        
        
        #agent 투입
        self.add_agent(step)               

        #action 적용
        traci.simulationStep()

        #agent의 생성이나 제거를 판단
        self.agent_update()
        
        #next_state 생성
        next_state = self.collect_state()
        
        #reward 생성
        reward = self.collect_reward()
        
        return next_state, reward,self.num_agent
    
    #check if agent can change to right lane
    def changeLaneRight(self, agent):
        changeLaneInfo = traci.vehicle.couldChangeLane(agent, -1)
        return changeLaneInfo

    #check if agent can change to left lane
    def changeLaneLeft(self, agent):
        changeLaneInfo = traci.vehicle.couldChangeLane(agent, 1)
        return changeLaneInfo   

    #Return distance from leading car, -1 if none
    def leader(self, agent):
        try:
            leadDistance = traci.vehicle.getLeader(agent, 0.0)[1]
        except TypeError:
            leadDistance = -1        
        return leadDistance

    #Return distance from following car, -1 if none
    def follower(self, agent):
        try:
            followDistance = traci.vehicle.getFollower(agent, 0.0)[1]
        except TypeError:
            followDistance = -1        
        return followDistance

    def get_observ_list(self):
        #observ = list()
        observ_list = [
             self.ChangeLaneRight, #whether agent can make lane change to right
             self.ChangeLaneLeft, #whether agent can make lane change to left
             traci.vehicle.getSpeed, #current speed of agent
             self.leader, #distance between leading car
             self.follower, #distance between following car
             traci.vehicle.getLaneIndex, #index of current lane
             traci.vehicle.getRouteIndex, #index of current edge
             0 #for 내 차량이 갈 방향
        ]
        return observ_list

    
    #move to left lane
    def actionLeftLane(self, agent):
        traci.vehicle.changeLaneRelative(agent, 1, 0)
    
    #move to right lane
    def actionRightLane(self, agent):
        traci.vehicle.changeLaneRelative(agent, -1, 0)
    
    #stay on current lane
    def actionStayLane(self, agent):
        traci.vehicle.changeLaneRelative(agent, 0, 0)
