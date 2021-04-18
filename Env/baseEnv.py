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
ENV_CONFIGS = {
    'state_space': 8,
    'gen_agent_list': ['agent_0', 'agent_1', 'agent_2', 'agent_3'],
    'action_size': 2  # 방향(좌/우) / 속도(가속/감속)
}


class Env():
    # __init__에서 반영하는 변수는 추후 config.py로 이동할것
    def __init__(self, configs):
        if configs['mode'] != 'load_train':
            configs['ENV_CONFIGS'] = ENV_CONFIGS
        self.env_configs = configs['ENV_CONFIGS']
        self.agent_list = list()
        self.gen_agent_list = self.env_configs['gen_agent_list']
        self.num_agent = 0
        self.state_space = self.env_configs['state_space']
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.reward = 0

        self.observ_list = self.get_observ_list()
        self.route_dict = dict()

    def init(self):
        state = self.collect_state()
        for id in traci.route.getIDList():
            self.route_dict[id] = traci.route.getEdges(id)

        return state, self.num_agent

    # gen_agent_list에 존재하는 agent를 50 timestep 단위로 투입후 agent_list에 추가
    def add_agent(self, step):
        for idx, agent in enumerate(self.gen_agent_list):
            if step == 50*idx:
                traci.vehicle.add(vehID=agent, routeID='route_0',
                                  typeID='rl_agent', departLane='random')
                self.agent_list.append(agent)
                self.num_agent += 1

    # agent의 생성과 제거를 판단

    def agent_update(self):
        # 도착한 agent 제거
        arrived_list = traci.simulation.getArrivedIDList()
        for idx, agent in enumerate(self.agent_list):
            if agent in arrived_list:
                self.agent_list.pop(idx)  # agent_list에서 도착 agent 제거
                self.num_agent -= 1

    def collect_state(self):
        next_state = torch.zeros(
            (self.num_agent, self.state_space), dtype=torch.float, device=self.device)
        agent_state = torch.zeros(
            (1, self.state_space), dtype=torch.float, device=self.device)  # agent_state는 agent별 state를 나타냄

        for i, agent in enumerate(self.agent_list):
            for idx, observ in enumerate(self.observ_list):
                agent_state[0, idx] = observ(agent)
                next_state[i, :] = agent_state.clone().detach()

        return next_state

    def collect_reward(self):
        reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        agent_reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        # cum_reward = torch.like(reward) cumulative reward 필요한가?

        for idx, agent in enumerate(self.agent_list):
            speed = traci.vehicle.getSpeed(agent)
            if speed < 0:
                speed = 0
            agent_reward[idx] = speed
            reward = agent_reward.clone().detach()
        return reward

    def collect_penalty(self):
        penalty = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        agent_penalty = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        # for idx, agent in enumerate(self.agent_list):
        #     # traci.vehicle.
        #     print("no")
        return penalty

    def step(self, action, step):
        # action mapping
        if self.num_agent != 0:
            for idx, agent in enumerate(self.agent_list):
                currentSpeed = traci.vehicle.getSpeed(agent)
                acc = action[idx, 0]
                print(acc)
                traci.vehicle.setSpeed(agent, currentSpeed+acc)

                self.changeLaneAction(agent, action[idx, 1])

        # agent 투입
        self.add_agent(step)

        # action 적용
        traci.simulationStep()

        # agent의 생성이나 제거를 판단
        self.agent_update()

        # next_state 생성
        next_state = self.collect_state()

        # reward 생성
        reward = self.collect_reward()
        # penalty 생성
        penalty = self.collect_penalty()
        self.reward += (reward.sum()-penalty.sum())

        return next_state, reward-penalty, self.num_agent

    # check if agent can change to right lane
    def changeLaneRight(self, agent):
        changeLaneInfo = traci.vehicle.couldChangeLane(agent, -1)
        return changeLaneInfo

    # check if agent can change to left lane
    def changeLaneLeft(self, agent):
        changeLaneInfo = traci.vehicle.couldChangeLane(agent, 1)
        return changeLaneInfo

    # Return distance from leading car, -1 if none
    def leader(self, agent):
        try:
            leadDistance = traci.vehicle.getLeader(agent, 0.0)[1]
        except TypeError:
            leadDistance = -1
        return leadDistance

    # Return distance from following car, -1 if none
    def follower(self, agent):
        try:
            followDistance = traci.vehicle.getFollower(agent, 0.0)[1]
        except TypeError:
            followDistance = -1
        return followDistance

    def get_observ_list(self):
        #observ = list()
        observ_list = [
            traci.vehicle.getSpeed,  # current speed of agent
            self.changeLaneRight,  # whether agent can make lane change to right
            self.changeLaneLeft,  # whether agent can make lane change to left
            self.leader,  # distance between leading car
            self.follower,  # distance between following car
            traci.vehicle.getLaneIndex,  # index of current lane
            traci.vehicle.getRouteIndex,  # index of current edge
            #self.get_direction  #for 내 차량이 갈 방향
        ]
        return observ_list

    def changeLaneAction(self, agent, laneChangeAction):
        if laneChangeAction-1 == 1:  # left
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0:
                return
            if lane[-1] == str(traci.edge.getLaneNumber(lane[:-2])-1):
                laneChangeAction = 0
            else:
                traci.vehicle.changeLaneRelative(agent, 1, 0)
        elif laneChangeAction-1 == 1:  # right
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0:
                return
            if lane[-1] == str(0):
                laneChangeAction = 0
            else:
                traci.vehicle.changeLaneRelative(agent, -1, 0)
        elif laneChangeAction-1 == 1:  # straight
            traci.vehicle.changeLaneRelative(agent, 0, 0)

    
    #direction은 [current edge]-[surrounding edge]-[probability]의 순으로 구성된 중첩 dictionary
    #junction_edges는 [junction(node)_id]-[surrounding edge]의 순으로 구성된 dictionary
    def get_direction(self, agent):
        junction_edges = get_junction_from_net_xml() #dict
        direction = dict()
        direction_key_sorted = list()

        cur_edge = traci.vehicle.getRoadID(agent)
        for index in junction_edges.keys():
            if cur_edge in junction_edges[index] #index는 junction_edges의 key가 되는 node 이름
                cur_node = index 
                direction[cur_edge] = junction_edges[cur_node] #juction을 이루는 node들
                
                direction_key_list = direction[cur_edge]
                for i in range(len(direction_key_list)):
                    if direction_key_list[i] = cur_edge:
                        calib_val = i 
                    if i-calib_val >= 0:
                        direction_key_sorted[i] = direction_key_list[i-calib_val] 
                    else:
                        direction_key_sorted[i] = direction_key_list[i+calib_val] #cur_edge가 [0]번째가 되도록 sort
                
                for idx, sur_edge in enumerate(direction_key_sorted[cur_edge]):
                    num_edges = len(direction[cur_edge])
                    direction[cur_edge][sur_edge] = (idx / num_edges)
            
                    next_edge = get_next_edge()
                    next_edge_val = direction[cur_edge][next_edge]
        
        return next_edge_val


    #map 내에 존재하는 모든 junction들에 대해 node_id를 key로, node를 둘러싼 edge를 값으로 갖는 딕셔너리를 반환
    def get_junction_from_net_xml(self):
        add_file_path = os.path.join(
            self.configs['current_path'], 'Network', self.configs['load_file_name']+'.net.xml')
        junction_edge = list()

        net_tree = parse(self.net_file_path)
        junctions = net_tree.findall('junction') #string
        junction_edge_dict = dict()
        
        for junction in junctions:
            if junction.attrib['type'] == "traffic_light": #map 밖으로 향하는 junction은 배제하고
                junction_id = junction.attrib['id'] #junction의 id 저장
                
                incLanes = junction.attrib['incLanes'] #incLanes의 긴 string에서
                incLanes = incLanes.split() #space 단위로 구성 edge를 split 후 리스트에 저장
            
                for edge in incLanes:
                    edge = edge[:-2] #끝부분 lane number 요소 제거하고
                    junction_edge.append(edge)

                junction_edge = list(dict.fromkeys(junction_edge))

                junction_edge_dict[junction_id] = junction_edge
        
        return junction_edge_dict


    #역전된 next_edge를 반환(junction을 이루는 edge는 모두 중심을 향하므로)
    def get_next_edge(self, agent):
        route = traci.vehicle.getRoute(agent):
        cur_edge = traci.vehicle.getRoadID(agent)
        
        for index in route:
            if route[index] == cur_edge:
                next_edge = route[index+1]
                
                from_edge = next_edge.split('to')[0]
                to_edge = next_edge.split('to')[1]
                next_edge_rev = "{} to {}".format(to_edge, from_edge)            
                
                if cur_edge == route[len(route)]: #cur_edge가 route의 마지막인 경우 None을 반환
                    next_edge_rev = None
        
        return next_edge_rev    
