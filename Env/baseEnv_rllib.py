'''
state size
1.changeLane 가능여부(좌/우)
2.앞차와의 거리
3.뒷차와의 거리
4.내 속도
5.내 레인 위치
6.내 edge상의 위치
7.내 차량이 갈 방향
8.내 edge의 신호등 상황
'''
'''
각 텐서 형태
state -> (agent갯수(1)) * (state갯수(8)) tensor
action -> (agent갯수(1) * (action_size(2)) tensor
reward -> (agent갯수(1)) * (1) tensor
'''

import copy
import random
import torch
import traci
import os
from xml.etree.ElementTree import parse
import ray
from ray.rllib.agents import ppo

ENV_CONFIGS = {
    'state_space': 9,
    'action_size': 2,
    'gen_agent_list': ['agent_{}'.format(i) for i in range(70)],
    'route_list':['route_{}'.format(i) for i in range(3)]
}

class Env():
    def __init__(self, file_path, device, configs):
        ##환경 설정
        if configs['mode'] != 'load_train':
            configs['ENV_CONFIGS'] = ENV_CONFIGS
        self.mode=configs['mode']
        self.device = device
        self.file_path = file_path        
        
        ##학습 설정
        self.env_configs = configs['ENV_CONFIGS']
        self.agent_list = list()
        self.gen_agent_list = self.env_configs['gen_agent_list']
        self.vehicle_gen_idx=0
        self.num_agent = 0
        self.state_space = self.env_configs['state_space']
        self.file_name = 'cross'
        self.route_list=self.env_configs['route_list']
        self.reward = 0
        self.observ_list = self.get_observ_list()
        self.route_dict = dict()
        self.agent_route_dict = dict()
        self.popup_action = None # action의 agent별 update 변화를 감당하는 action


    def init(self):
        state = self.collect_state()
        for id in traci.route.getIDList():
            self.route_dict[id] = traci.route.getEdges(id)

        return state, self.num_agent

    '''
    #reset 부는 rllib 구조상 필요할 수 있어서 추후에 추가 예정
    def reset(self):
        state = dict()
        return state
    '''

    def step(self, action, step):
        # action mapping
        if self.num_agent != 0:
            self.popup_action=action.detach().clone()
            for idx, agent in enumerate(self.agent_list):
                currentSpeed = traci.vehicle.getSpeed(agent)
                acc = action[idx, 0]
                traci.vehicle.setSpeed(agent, currentSpeed+acc)
                self.changeLaneAction(agent, int(action[idx, 1]))

        # agent 투입
        self.add_agent(step)

        # action 적용
        traci.simulationStep()

        # agent의 생성이나 제거를 판단
        self.agent_update()

        # next_state 생성
        state = self.collect_state()

        # reward 생성
        reward = self.calculate_reward()

        return state, reward, self.num_agent


    # gen_agent_list에 존재하는 agent를 50 timestep 단위로 투입후 agent_list에 추가
    def add_agent(self, step):
        if step >= float(50*self.vehicle_gen_idx) and self.vehicle_gen_idx < len(self.gen_agent_list):
            random.shuffle(self.route_list)
            if self.mode in ['train','load_train','test']:
                traci.vehicle.add(vehID=self.gen_agent_list[self.vehicle_gen_idx], routeID=self.route_list[0],
                                    typeID='rl_agent', departLane='random')
            elif self.mode =='simulate':#'non_rl_'
                traci.vehicle.add(vehID=self.gen_agent_list[self.vehicle_gen_idx], routeID=self.route_list[0],
                                    typeID='rl_agent', departLane='random')
            else:
                raise NotImplementedError
            self.agent_list.append(self.gen_agent_list[self.vehicle_gen_idx])
            self.num_agent += 1
            
            self.agent_route_dict[self.gen_agent_list[self.vehicle_gen_idx]] = self.route_list[0]

            add_tensor=torch.zeros((1,2),device=self.device,dtype=torch.int)
            if self.num_agent==1:
                self.popup_action = add_tensor.clone()
            else:
                self.popup_action = torch.cat([self.popup_action,add_tensor],dim=0)
            self.vehicle_gen_idx+=1

    # agent의 제거를 판단
    def agent_update(self):
        # 도착한 agent 제거
        arrived_list = traci.simulation.getArrivedIDList()
        agent_list=copy.deepcopy(self.agent_list)
        # tmp_num_agent=copy.deepcopy(self.num_agent)

        mask_idx=torch.ones(self.num_agent,dtype=torch.bool).view(-1)
        for idx, agent in enumerate(agent_list):
            if agent in arrived_list:
                self.agent_list.remove(agent) # agent_list에서 도착 agent 제거
                mask_idx[idx]=False
        
        self.num_agent=len(self.agent_list)
        if self.popup_action is None: # before action setting
            return
        else:
            self.popup_action=self.popup_action[mask_idx].detach().clone()


    def collect_state(self):
        state = dict()
        agent_state = list()
        
        for agent in self.agent_list:
            for observ in self.oberv_list:
                agent_state.append(observ(agent))
                state[agent] = agent_state
        return state

    def collect_pos_reward(self):
        if len(self.agent_list) == 0:
            return None
        
        pos_reward = dict()
        for agent in self.agent_list:
            pos_reward[agent] = max(traci.vehicle.getSpeed(agent), 0.0)       
        
        return pos_reward
    
    def collect_neg_reward(self, prev_lane): #prev_lane은 main()에 위치한 변수, utils.py의 eval_set_num_lane_change()에서 활용
        if len(self.agent_list) == 0:
            return None
        
        neg_reward = dict()
        
        #lanechange
        current_lane = torch.zeros(
            (len(self.gen_agent_list), 1), dtype=torch.float, device=self.device)

        for idx, agent in enumerate(self.gen_agent_list):
            if agent in self.agent_list:
                current_lane[idx] = traci.vehicle.getLaneIndex(agent)
        
        diff = torch.eq(current_lane, prev_lane)

        for i, agent in enumerate(self.gen_agent_list):
            if diff[i] == False:
                neg_reward[agent] += 1.0

        #current_lane tensor를 prev_lane으로 copy할 시 for문을 이용하지 않으면 전역변수임에도 불구하고 제대로 process 되지 않는 문제가 발생
        for i in range (len(self.gen_agent_list)):
            prev_lane[i] = current_lane[i] 

        #teleport
        teleport_list = traci.simulation.getStartingTeleportIDList()
        for teleport_veh in teleport_list:
            if teleport_veh in self.agent_list:
                idx = self.agent_list.index(teleport_veh)
                lane = traci.vehicle.getLaneID(teleport_veh)
                if len(lane) == 0: # error
                    neg_reward[teleport_veh] += 5.0
                else:
                    neg_reward[teleport_veh] += traci.lane.getMaxSpeed(lane)/2.0
        return neg_reward

    def calculate_reward(self):
        pos_reward = self.collect_pos_reward()
        neg_reward = self.collect_neg_reward()
        reward = dict()
        
        for agent in self.agent_list: 
            reward[agent] += pos_reward[agent]-neg_reward[agent]
        return reward


##Observation related functions
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
            leadDistance = min(traci.vehicle.getLeader(agent, 0.0)[1],100)
        except TypeError:
            leadDistance = 100.0
        return leadDistance/100.0

    # Return distance from following car, -1 if none
    def follower(self, agent):
        try:
            followDistance = min(traci.vehicle.getFollower(agent, 0.0)[1],100)
        except TypeError:
            followDistance = 100.0
        return followDistance/100.0

    def get_observ_list(self):
        #observ = list()
        observ_list = [
            self.getSpeed,  #current speed of agent
            self.changeLaneRight,  #whether agent can make lane change to right
            self.changeLaneLeft,  #whether agent can make lane change to left
            self.leader,  #distance between leading car
            self.follower,  #distance between following car
            traci.vehicle.getLaneIndex,  #index of current lane
            traci.vehicle.getRouteIndex,  #index of current edge
            self.get_direction,  #direction of agent
            self.get_traffic_light #traffic light status of current edge
        ]
        return observ_list
    
    def getSpeed(self,agent):
        velocity=traci.vehicle.getSpeed(agent)
        return max(velocity,0.0)

    def changeLaneAction(self, agent, laneChangeAction):
        if laneChangeAction-1 == 1:  # left
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0: # no lane
                return
            if lane[-1] == str(traci.edge.getLaneNumber(lane[:-2])-1):
                return
            else:
                traci.vehicle.changeLaneRelative(agent, 1, 0)
        elif laneChangeAction-1 == -1:  # right
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0:# no lane
                return
            if lane[-1] == str(0):
                return
            else:
                traci.vehicle.changeLaneRelative(agent, -1, 0)
        elif laneChangeAction-1 == 0:  # straight
            # traci.vehicle.changeLaneRelative(agent, 0, 0)
            return
        else:
            raise NotImplementedError


    # direction은 [current edge]-[surrounding edge]-[probability]의 순으로 구성된 중첩 dictionary
    # junction_edges는 map의 모든 junction에 대해 [junction(node)_id]-[surrounding edge]의 순으로 구성된 dictionary

    def get_direction(self, agent):
        junction_edges = self.get_junction_from_net_xml()
        direction = dict()
        direction_key_sorted = list()
        next_edge_val = 0.5
        cur_edge = self.get_cur_edge(agent)

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

                    next_edge = self.get_next_edge(cur_edge, agent)
                    if next_edge != None:
                        next_edge_val = direction[cur_edge][next_edge]
                    else:
                        next_edge_val = 0.5
                        
            #print('age-route:', agent, self.agent_route_dict[agent])
            #print('curr_edge:', agent, cur_edge)
            #print('next_valu:', agent, next_edge_val)
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

    def get_next_edge(self, cur_edge, agent):
        route = self.route_dict[self.agent_route_dict[agent]]  # list

        idx = route.index(cur_edge)

        if idx != len(route)-1:
            next_edge_rev = route[idx + 1]

            # junction을 이루는 edge는 모두 중심을 향하므로 agent가 나아갈 next_edge는 node가 반전
            from_edge = next_edge_rev.split('_to_')[0]
            to_edge = next_edge_rev.split('_to_')[1]
            next_edge = "{}_to_{}".format(to_edge, from_edge).strip()
        else:
            next_edge = None

        return next_edge
    
    def get_cur_edge(self, agent):
        index = max(traci.vehicle.getRouteIndex(agent), 0)        
        cur_edge = self.route_dict[self.agent_route_dict[agent]][index]
        
        return cur_edge
    
    
    def get_traffic_light(self, agent):
        cur_edge = self.get_cur_edge(agent)
        next_node = 'n_' + cur_edge.split('_to_')[1]

        if next_node in traci.trafficlight.getIDList():
            tl_state = self.mapping_tl(cur_edge, next_node)
        else:
            tl_state = -1 #not exist
        # print(next_node,tl_state)
        return tl_state

    def mapping_tl(self, cur_edge, next_node):
        tl_dict = dict()
        
        all_tl = traci.trafficlight.getRedYellowGreenState(next_node)
        tl_dict['U_to_C'] = all_tl[0:5].lower()
        tl_dict['R_to_C'] = all_tl[5:10].lower()
        tl_dict['D_to_C'] = all_tl[10:15].lower()
        tl_dict['L_to_C'] = all_tl[15:20].lower() #실제 사용될 tl



        if tl_dict[cur_edge] == 'rrrrr': #정지
            tl_state = 0
        elif tl_dict[cur_edge] == 'yyyyy': #대기
            tl_state = 1
        elif tl_dict[cur_edge] == 'grrrg': #좌/우회전
            tl_state = 2
        elif tl_dict[cur_edge] == 'ggggr': #직진
            tl_state = 3        
        
        return tl_state 
    