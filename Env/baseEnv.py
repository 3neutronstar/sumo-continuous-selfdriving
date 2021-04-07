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
1.이전 action을 반환
2.next state 계산
3.
'''
import torch
import traci
from copy import deepcopy
from configs import EXP_CONFIGS

#config로 이동할것
state_space = 8
agent_list = ['agent_0']
device = 'cpu'
action_size = 2 # 방향(좌/우) / 속도(가속/감속)


class Env():
    #__init__에서 반영하는 변수는 추후 config.py로 이동할것
    def __init__(self, configs):
        self.configs = configs
        self.device = device
        self.agent_list = agent_list
        self.num_agent = len(agent_list)

        self.reward = torch.zeros((1, self.num_agent), dtype=torch.float, device=device)
        self.cum_reward = torch.zeros_like(self.reward)
        self.state_space = state_space
  
    '''
    def get_action(self, mask):
        action = torch.zeros(
            (self.num_agent, self.action_size), dtype=torch.float, device=device)

        for idx in torch.nonzero(mask):
            action[idx,:] = deepcopy(self.tl_rl_memory[index].action)

        return action
    '''  
    
    def collect_state(self):         
        self.agent_list = agent_list
        self.num_agent = len(self.agent_list)
        self.observ_list = self.observ_list()
        
        next_state = torch.zeros(
            (self.num_agent, self.state_space), dtype=torch.float, device=device)
        agent_state = torch.zeros(
            (1, num_observ), dtype=torch.float, device=device) #agent_state는 agent별 state를 나타냄
        
        for _, agent in enumerate(self.agent_list):
            for idx in enumerate(self.observ_list):
                agent_state[0, idx] = observ[idx](agent)
                next_state[_,:] = next_state.clone().detach()

        return next_state


    def collect_reward(self):
        self.agent_list = agent_list
        self.num_agent = len(self.agent_list)
        
        reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=device)
        agent_reward = torch.zeros(
            (1, 1), detype=torch.float, device=device)
        #cum_reward = torch.like(reward) cumulative reward 필요한가?
        
        for _, agent in enumerate(self.agent_list):           
            agent_reward[0, 0] = traci.vehicle.getSpeed(agent)
            reward = agent_reward.clone().detach()

        #self.cum_reward += reward        
        return reward      
   
    def step(self, action):
        #agent로부터 action tensor를 반환받을것
        action = torch.zeros(
            (self.num_agent, action_size), dtype=torch.float, device=device)
        
        #action mapping
        for _, agent in torch.nonzero(mask_matrix):
            currentSpeed = traci.vehicle.getSpeed(agent)
            acc = action[_, 0]
            traci.vehicle.setSpeed(agent, currentSpeed+acc)
            
            if action[_, 1] == 1:
                self.actionLeftLane(agent)
            elif action[_, 1] == -1:
                self.actionRightLane(agent)
            else:
                self.actionStayLane(agent)

        #action 적용
        traci.simulationStep()

        #next_state 생성
        next_state = self.collect_state()
        
        #reward 생성
        reward = self.collect_reward()
        
        return next_state, reward
    
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

    def observ_list(self):
        #observ = list()
        observ = [
             ChangeLaneRight, #whether agent can make lane change to right
             ChangeLaneLeft, #whether agent can make lane change to left
             traci.vehicle.getSpeed, #current speed of agent
             leader, #distance between leading car
             follower, #distance between following car
             traci.vehicle.getLaneIndex, #index of current lane
             traci.vehicle.getRouteIndex, #index of current edge
             0 #for 내 차량이 갈 방향
        ]
        return observ

    
    #move to left lane
    def actionLeftLane(self, agent):
        traci.vehicle.changeLaneRelative(agent, 1, 0)
    
    #move to right lane
    def actionRightLane(self, agent):
        traci.vehicle.changeLaneRelative(agent, -1, 0)
    
    #stay on current lane
    def actionStayLane(self, agent):
        traci.vehicle.changeLaneRelative(agent, 0, 0)
