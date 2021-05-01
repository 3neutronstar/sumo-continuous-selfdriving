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
import copy
import random
import torch
import traci
import os
from xml.etree.ElementTree import parse
ENV_CONFIGS = {
    'state_space': 8,
    'gen_agent_list': ['agent_{}'.format(i) for i in range(70)],
    'route_list':['route_{}'.format(i) for i in range(3)],
    'action_size': 2
}


class Env():
    def __init__(self, file_path, device, configs):
        if configs['mode'] != 'load_train':
            configs['ENV_CONFIGS'] = ENV_CONFIGS
        self.env_configs = configs['ENV_CONFIGS']
        self.agent_list = list()
        self.gen_agent_list = self.env_configs['gen_agent_list']
        self.vehicle_gen_idx=0

        self.num_agent = 0
        self.state_space = self.env_configs['state_space']
        self.device = device
        self.file_path = file_path
        self.file_name = 'cross'
        self.route_list=self.env_configs['route_list']

        self.reward = 0

        self.observ_list = self.get_observ_list()
        self.route_dict = dict()
        self.prev_lane_idx=None

    def init(self):
        state = self.collect_state()
        for id in traci.route.getIDList():
            self.route_dict[id] = traci.route.getEdges(id)

        return state, self.num_agent

    # gen_agent_list에 존재하는 agent를 50 timestep 단위로 투입후 agent_list에 추가
    def add_agent(self, step):
        if step >= float(50*self.vehicle_gen_idx) and self.vehicle_gen_idx < len(self.gen_agent_list):
            random.shuffle(self.route_list)
            traci.vehicle.add(vehID=self.gen_agent_list[self.vehicle_gen_idx], routeID=self.route_list[0],
                                typeID='rl_agent', departLane='random')
            self.agent_list.append(self.gen_agent_list[self.vehicle_gen_idx])
            self.num_agent += 1
            add_tensor=torch.zeros((1,1),device=self.device,dtype=torch.int)
            if self.num_agent==1:
                self.prev_lane_idx=add_tensor
            else:
                self.prev_lane_idx=torch.cat([self.prev_lane_idx,add_tensor],dim=0)
            self.vehicle_gen_idx+=1

    # agent의 생성과 제거를 판단
    def agent_update(self):
        # 도착한 agent 제거
        arrived_list = traci.simulation.getArrivedIDList()
        agent_list=copy.deepcopy(self.agent_list)
        for idx, agent in enumerate(agent_list):
            if agent in arrived_list:
                self.agent_list.remove(agent) # agent_list에서 도착 agent 제거
                self.num_agent -= 1
                if idx==0:
                    self.prev_lane_idx=self.prev_lane_idx[:,1:]
                if idx==self.num_agent:
                    self.prev_lane_idx=self.prev_lane_idx[:,:-1]
                elif self.prev_lane_idx.size()[0]>1:
                    self.prev_lane_idx=torch.cat([self.prev_lane_idx[:idx,:],self.prev_lane_idx[idx+1:,:]],dim=0)
                else:
                    self.prev_lane_idx=None


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
        current_lane = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        for idx, agent in enumerate(self.agent_list):
            current_lane[idx]=traci.vehicle.getLaneIndex(agent)
        pen=torch.eq(current_lane,self.prev_lane_idx).view(-1,1)
        if pen.size()[0]!=0:
            penalty[pen]-=1.0
        self.prev_lane_idx=current_lane.clone()
        #     print("no")
        return penalty

    def step(self, action, step):
        # action mapping
        if self.num_agent != 0:
            for idx, agent in enumerate(self.agent_list):
                currentSpeed = traci.vehicle.getSpeed(agent)
                acc = action[idx, 0]
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
            self.get_direction  # for 내 차량이 갈 방향
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

    # direction은 [current edge]-[surrounding edge]-[probability]의 순으로 구성된 중첩 dictionary
    # junction_edges는 map의 모든 junction에 대해 [junction(node)_id]-[surrounding edge]의 순으로 구성된 dictionary

    def get_direction(self, agent):
        junction_edges = self.get_junction_from_net_xml()
        direction = dict()
        direction_key_sorted = list()
        next_edge_val = 0.5

        # getRoadID의 반환값인 cur_edge가 왜 튜플인지?
        cur_edge = traci.vehicle.getRoadID(agent)
        #cur_edge = ''.join(cur_edge_tuple)
        for cur_node in junction_edges.keys():  # 모든 junction node에 대해, index는 junction_edges의 key가 되는 node의 id
            # cur_edge가 junction을 구성하는 edge 중 하나라면
            if cur_edge in junction_edges[cur_node]:
                direction[cur_edge] = dict.fromkeys(junction_edges[cur_node])

                # [sur_edge]가 clockwise order를 유지하며 [cur_edge]가 [0]번째 위치 하도록 sort
                direction_key_list = list(direction[cur_edge])
                for i in range(len(direction_key_list)):
                    for j in range(len(direction_key_list)):
                        if direction_key_list[j] == cur_edge:
                            calib_val = j

                    if calib_val+i < len(direction_key_list):
                        direction_key_sorted.append(
                            direction_key_list[calib_val+i])
                    else:
                        direction_key_sorted.append(
                            direction_key_list[calib_val+i-len(direction_key_list)])

                # direction을 분수로 표현, cur_edge가 0이 되며, 시계방향으로 숫자가 커지도록
                # 즉, 0에 가까운 숫자일수록 좌회전, 1에 가까운 숫자일수록 우회전으로 인지될 수 있도록
                ######
                for idx, sur_edge in enumerate(direction_key_sorted):
                    num_edges = len(direction[cur_edge])
                    direction[cur_edge][sur_edge] = (idx / num_edges)

                    next_edge = self.get_next_edge(cur_edge)
                    if next_edge != None:
                        next_edge_val = direction[cur_edge][next_edge]
                    else:
                        next_edge_val = 0.5
        return next_edge_val

    # map 내에 존재하는 모든 junction들에 대해 node_id를 key로, node를 둘러싼 edge_id의 list를 값으로 갖는 딕셔너리를 반환

    def get_junction_from_net_xml(self):
        add_file_path = os.path.join(
            self.file_path, 'Net_data', self.file_name + '.net.xml')
        net_tree = parse(add_file_path)
        # junction attribute를 .net.xml에서 반환
        junctions = net_tree.findall('junction')

        junction_edge = list()
        junction_edge_dict = dict()

        for junction in junctions:
            if junction.attrib['type'] == "traffic_light":  # map 밖으로 향하는 junction은 배제하고
                junction_id = junction.attrib['id']  # junction의 id 저장

                # incLanes이 가리키는 일련의 긴 string에서
                incLanes = junction.attrib['incLanes']
                incLanes = incLanes.split()  # space 단위로 구성 edge를 split 후 리스트에 저장

                for edge in incLanes:
                    edge = edge[:-2]  # 끝자리 lane_num 요소(eg: _0) 제거하고
                    junction_edge.append(edge)

                junction_edge = list(dict.fromkeys(
                    junction_edge))  # 중복 제거 후 list화

                junction_edge_dict[junction_id] = junction_edge

        return junction_edge_dict  # list를 value로 갖는 딕셔너리 반환

    # next_edge를 반환

    def get_next_edge(self, cur_edge):
        route = self.route_dict['route_0']  # list

        for idx in range(len(route)):
            if route[idx] == cur_edge:  # current_edge가 route 내의 몇번째에 위치하는지 idx로 반환
                next_edge_rev = route[idx + 1]

                # junction을 이루는 edge는 모두 중심을 향하므로 agent가 나아갈 next_edge는 node가 반전
                from_edge = next_edge_rev.split('_to_')[0]
                to_edge = next_edge_rev.split('_to_')[1]
                next_edge = "{}_to_{}".format(to_edge, from_edge).strip()

                if cur_edge == route[len(route)-1]:
                    next_edge = None  # cur_edge가 route의 마지막인 경우 None을 return

        return next_edge
