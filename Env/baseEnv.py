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
        self.mode=configs['mode']
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
        self.agent_route_dict = dict()
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
            if self.mode in ['train','load_train','test']:
                traci.vehicle.add(vehID=self.gen_agent_list[self.vehicle_gen_idx], routeID=self.route_list[0],
                                    typeID='rl_agent', departLane='random')
            elif self.mode =='simulate':
                traci.vehicle.add(vehID='non_rl_'+self.gen_agent_list[self.vehicle_gen_idx], routeID=self.route_list[0],
                                    typeID='rl_agent', departLane='random')
            else:
                raise NotImplementedError
            self.agent_list.append(self.gen_agent_list[self.vehicle_gen_idx])
            self.num_agent += 1
            add_tensor=torch.zeros((1,1),device=self.device,dtype=torch.int)
            
            self.agent_route_dict[self.gen_agent_list[self.vehicle_gen_idx]] = self.route_list[0]
            if self.num_agent==1:
                self.prev_lane_idx=add_tensor.clone()
            else:
                self.prev_lane_idx=torch.cat([self.prev_lane_idx,add_tensor],dim=0)
            self.vehicle_gen_idx+=1

    # agent의 생성과 제거를 판단
    def agent_update(self):
        # 도착한 agent 제거
        arrived_list = traci.simulation.getArrivedIDList()
        agent_list=copy.deepcopy(self.agent_list)
        # tmp_num_agent=copy.deepcopy(self.num_agent)

        mask_idx=torch.ones_like(self.prev_lane_idx,dtype=torch.bool).view(-1)
        for idx, agent in enumerate(agent_list):
            if agent in arrived_list:
                self.agent_list.remove(agent) # agent_list에서 도착 agent 제거
                mask_idx[idx]=False
                # self.num_agent -= 1
        self.num_agent=mask_idx.sum()
        self.prev_lane_idx=self.prev_lane_idx[mask_idx].detach().clone()

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
        if len(self.agent_list)==0:
            return None
        reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        agent_reward = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        # cum_reward = torch.like(reward) cumulative reward 필요한가?

        for idx, agent in enumerate(self.agent_list):
            velocity = traci.vehicle.getSpeed(agent)
            agent_reward[idx] = max(velocity,0.0)
        reward = agent_reward.detach().clone()
        return reward
    
    def collect_penalty(self):
        if len(self.agent_list)==0:
            return None
        # lanechange penalty
        penalty = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        current_lane = torch.zeros(
            (self.num_agent, 1), dtype=torch.float, device=self.device)
        for idx, agent in enumerate(self.agent_list):
            current_lane[idx]=traci.vehicle.getLaneIndex(agent)
        pen=torch.eq(current_lane,self.prev_lane_idx).view(-1,1)
        if pen.size()[0]!=0: #0개의 size를 가지고 있지 않다면
            penalty[~pen]+=1.0
        self.prev_lane_idx=current_lane.clone()

        # teleport penalty
        teleport_list=traci.simulation.getStartingTeleportIDList()
        for tp_veh in teleport_list:
            if tp_veh in self.agent_list:
                idx=self.agent_list.index(tp_veh)
                lane=traci.vehicle.getLaneID(tp_veh)
                if len(lane)==0: # error
                    penalty[idx]+=5.0
                else:
                    penalty[idx]+=traci.lane.getMaxSpeed(lane)/2.0
        return penalty.detach().clone()

    def step(self, action, step):
        # action mapping
        if self.num_agent != 0:
            for idx, agent in enumerate(self.agent_list):
                currentSpeed = traci.vehicle.getSpeed(agent)
                acc = action[idx, 0]
                traci.vehicle.setSpeed(agent, currentSpeed+acc)
                action[idx,1]=self.changeLaneAction(agent, int(action[idx, 1]))

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
        if penalty==None and reward ==None :
            return_reward=torch.tensor([0.0],dtype=torch.float,device=self.device)
        else:
            self.reward += (reward.sum()-penalty.sum())
            return_reward=reward-penalty
            

        return next_state, return_reward, self.num_agent

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
            leadDistance = -1.0
        return leadDistance

    # Return distance from following car, -1 if none
    def follower(self, agent):
        try:
            followDistance = traci.vehicle.getFollower(agent, 0.0)[1]
        except TypeError:
            followDistance = -1.0
        return followDistance

    def get_observ_list(self):
        #observ = list()
        observ_list = [
            self.getSpeed,  # current speed of agent
            self.changeLaneRight,  # whether agent can make lane change to right
            self.changeLaneLeft,  # whether agent can make lane change to left
            self.leader,  # distance between leading car
            self.follower,  # distance between following car
            traci.vehicle.getLaneIndex,  # index of current lane
            traci.vehicle.getRouteIndex,  # index of current edge
            self.get_direction  # for 내 차량이 갈 방향
        ]
        return observ_list
    
    def getSpeed(self,agent):
        velocity=traci.vehicle.getSpeed(agent)
        return max(velocity,0.0)

    def changeLaneAction(self, agent, laneChangeAction):
        if laneChangeAction-1 == 1:  # left
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0: # no lane
                return 1.0
            if lane[-1] == str(traci.edge.getLaneNumber(lane[:-2])-1):
                laneChangeAction = 1.0
            else:
                traci.vehicle.changeLaneRelative(agent, 1, 0)
        elif laneChangeAction-1 == -1:  # right
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0:# no lane
                return 1.0
            if lane[-1] == str(0):
                laneChangeAction = 1.0
            else:
                traci.vehicle.changeLaneRelative(agent, -1, 0)
        elif laneChangeAction-1 == 0:  # straight
            # traci.vehicle.changeLaneRelative(agent, 0, 0)
            pass
        else:
            raise NotImplementedError
        return float(laneChangeAction)


    # direction은 [current edge]-[surrounding edge]-[probability]의 순으로 구성된 중첩 dictionary
    # junction_edges는 map의 모든 junction에 대해 [junction(node)_id]-[surrounding edge]의 순으로 구성된 dictionary

    def get_direction(self, agent):
        junction_edges = self.get_junction_from_net_xml()
        direction = dict()
        direction_key_sorted = list()
        next_edge_val = 0.5

        # getRoadID의 반환값인 cur_edge가 왜 튜플인지?
        #cur_edge = traci.vehicle.getRoadID(agent)
        #print(agent, self.route_dict[self.agent_route_dict[agent]])
        #print(agent, traci.vehicle.getRouteIndex(agent))
        index = max(traci.vehicle.getRouteIndex(agent), 0)
        #print(agent, index)
        cur_edge = self.route_dict[self.agent_route_dict[agent]][index]

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
    
    def get_current_edge(self,agent):
        edge=traci.vehicle.getRoadID(agent)
        return edge

    def get_next_node(self,agent,edge):
        next_node='n_'+edge.split('_to_')[1]
        return next_node
    
    def get_traffic_light(self,agent):
        current_edge=self.get_current_edge(agent)
        next_node=self.get_next_node(agent,current_edge)
        if next_node in traci.trafficlight.getIDList():
            tl_state=self.mapping_tl(next_node)
        else:
            tl_state=-1#not exist
        
        return tl_state

    