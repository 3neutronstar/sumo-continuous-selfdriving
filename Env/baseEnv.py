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
import numpy as np
import copy
import random
import torch
import traci
import os
from xml.etree.ElementTree import parse
ENV_CONFIGS = {
    'state_space': 16,# 9+5+1+1, target velocity
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
        self.popup_action=None # action의 agent별 update 변화를 감당하는 action
        self.num_lane,self.num_edge = self.get_edge_from_edg_xml()

        self.cum_lanechange_num=dict()
        self.last_edge=dict()
        

    def init(self):
        removed_agent_idx=[]
        state = self.collect_state(removed_agent_idx)
        for id in traci.route.getIDList():
            self.route_dict[id] = traci.route.getEdges(id)
        done=None
        return state, done,self.num_agent

    
    def step(self, action, step):
        cur_speed=None
        # action mapping
        if self.num_agent != 0:
            if action!=[]:
                self.popup_action=action.detach().clone().numpy()
            cur_speed=[]
            for idx, agent in enumerate(self.agent_list):
                currentSpeed = traci.vehicle.getSpeed(agent)
                # acc = (action[idx, 0]-5.0)/2.0
                acc = action[idx, 0]#ddqn ddpg
                traci.vehicle.setSpeed(agent, currentSpeed+acc)
                self.changeLaneAction(agent, int(action[idx, 1]))
                cur_speed.append(currentSpeed)

        # agent 투입
        self.add_agent(step)
        done=np.zeros((self.num_agent,1),dtype=np.bool) 

        # action 적용
        traci.simulationStep()
        # agent의 생성이나 제거를 판단
        removed_agent_idx=self.agent_update()

        # next_state 생성
        next_state = self.collect_state(removed_agent_idx)
        
        #reward 생성
        reward = self.calculate_reward(removed_agent_idx)
        # print('reward:', reward)
        #action 보정
        if self.num_agent != 0 and cur_speed is not None:
            aft_speed=[]
            for idx, agent in enumerate(self.agent_list):
                if idx in removed_agent_idx:
                    continue
                afterSpeed = traci.vehicle.getSpeed(agent)
                aft_speed.append(afterSpeed)
            # print(count,action[count:],cur_speed,aft_speed)
            count=0
            for idx,a in enumerate(action):
                if idx in removed_agent_idx:
                    count+=1
                if idx-count>=0:
                    a[0]=min(max(round(((aft_speed[idx-count]-cur_speed[idx])*2)+5.0),0),8)
            # print(cur_speed,aft_speed,action[:,0].view(-1))
        
        #agent elimination check
        if 'fin' in self.agent_list:
            for idx in removed_agent_idx:
                self.agent_list.remove('fin')
                done[idx]=1
            self.num_agent=len(self.agent_list)
        else:
            done=np.zeros((self.num_agent))
        return torch.from_numpy(next_state), torch.from_numpy(reward), torch.from_numpy(done),self.num_agent


    """functions used in env steps"""
    #agent의 lanechange action을 결정
    def changeLaneAction(self, agent, laneChangeAction):
        if laneChangeAction-1 == 1:  # left
            lane = traci.vehicle.getLaneID(agent)
            if len(lane) == 0: # no lane
                return
            if lane[-1] == str(traci.edge.getLaneNumber(lane[:-2])-1):
                return
            else:
                traci.vehicle.changeLaneRelative(agent, 1, 0)
        elif (laneChangeAction-1) == -1:  # right
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


    #gen_agent_list에 존재하는 agent를 50 timestep 단위로 투입후 agent_list에 추가
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
            self.last_edge[self.gen_agent_list[self.vehicle_gen_idx]]='L_to_C'
            self.cum_lanechange_num[self.gen_agent_list[self.vehicle_gen_idx]]=0
            self.num_agent += 1
            
            self.agent_route_dict[self.gen_agent_list[self.vehicle_gen_idx]] = self.route_list[0]

            add_tensor=np.zeros((1,2),dtype=np.int)
            if self.num_agent==1:
                self.popup_action=copy.deepcopy(add_tensor)
            else:
                self.popup_action=np.concatenate([self.popup_action,add_tensor],axis=0)
            self.vehicle_gen_idx+=1

    #map 내에 존재하는 agent를 제거를 판단
    def agent_update(self)->list:
        # 도착한 agent 제거
        arrived_list = traci.simulation.getArrivedIDList()
        agent_list=copy.deepcopy(self.agent_list)
        remove_idx=[]

        mask_idx=np.ones(self.num_agent,dtype=np.bool).reshape(-1)
        for idx, agent in enumerate(agent_list):
            if agent in arrived_list:
                self.agent_list[idx]='fin' # agent_list에서 도착 agent 제거
                mask_idx[idx]=False
                remove_idx.append(idx)
        
        if self.popup_action is None: # before action setting
            return []
        else:
            self.popup_action=self.popup_action[mask_idx]
            return remove_idx


    def collect_state(self,removed_agent_idx):
        next_state = np.zeros(
            (self.num_agent, self.state_space), dtype=np.float32)
        for idx, agent in enumerate(self.agent_list):
            if idx in removed_agent_idx:
                continue

            this_agent_list=[]
            for i, observ in enumerate(self.observ_list):
                this_agent_list.append(np.array(observ(agent)).reshape(-1))

            agent_state = np.concatenate(this_agent_list,axis=0).reshape(-1)
            next_state[idx, :] = agent_state
        return next_state


    def collect_pos_reward(self,removed_agent_idx):
        if len(self.agent_list)==0:
            return None
        
        pos_reward = np.zeros(
            (self.num_agent, 1), dtype=np.float32)

        for idx, agent in enumerate(self.agent_list):
            if idx in removed_agent_idx:
                continue
            # speed = traci.vehicle.getSpeed(agent)
            # pos_reward[idx] = max(speed, 0.0)
            target_velocity=self.getDesiredVelocity(agent)
            pos_reward[idx]=self._calc_desired_velocity_reward(agent,target_velocity)
        return pos_reward
    
    def getDesiredVelocity(self,agent):
        # 신호를 받고
        cur_edge = self.get_cur_edge(agent)
        next_node = 'n_' + cur_edge.split('_to_')[1]

        tl_dict=dict()
        if next_node in traci.trafficlight.getIDList():
            all_tl = traci.trafficlight.getRedYellowGreenState(next_node)
            tl_dict['U_to_C'] = all_tl[0:5].lower()
            tl_dict['R_to_C'] = all_tl[5:10].lower()
            tl_dict['D_to_C'] = all_tl[10:15].lower()
            tl_dict['L_to_C'] = all_tl[15:20].lower() #실제 사용될 tl

        # 구간 최대 속도
        max_speed=traci.lane.getMaxSpeed(cur_edge+'_0')
        # 남은 거리 계산
        distance=self.getRestDistance(agent)
        leader_distance=self.getLeader(agent)

        if leader_distance < 0.5: # 장애물이 가까우면 desired velocity가 낮게
            desired_velocity=max_speed*leader_distance
            return desired_velocity

        if cur_edge not in tl_dict.keys():
            desired_velocity=max_speed
            return desired_velocity
        elif tl_dict[cur_edge] in ['rrrrr','yyyyy']: #정지 'grrrg'
            # 남은 거리기반 linear 감소
            if distance<0.3:
                desired_velocity=distance/0.3*max_speed
                return desired_velocity
        desired_velocity=max_speed
        return desired_velocity
        
    def _calc_desired_velocity_reward(self,agent,target_velocity):
        speed = traci.vehicle.getSpeed(agent)
        reward=max(target_velocity-abs(target_velocity-speed),0)/(target_velocity+1e-12)
        return reward

    def collect_neg_reward(self,removed_agent_idx):
        if len(self.agent_list)==0:
            return None
        
        #related to lanechange penalty
        neg_reward = np.zeros(
            (self.num_agent, 1), dtype=np.float32)
        if self.popup_action is None:
            pass
        else:
            for idx,action in enumerate(self.popup_action):
                if idx in removed_agent_idx:
                    continue
                lanechange_action=action[1]
                cur_edge=self.get_cur_edge(self.agent_list[idx])
                if self.last_edge[self.agent_list[idx]]!=cur_edge:#다른 edge로 가면 초기화
                    self.cum_lanechange_num[self.agent_list[idx]]=0
                self.last_edge[self.agent_list[idx]]=cur_edge
                if lanechange_action!=1:
                    smooth_val=5
                    self.cum_lanechange_num[self.agent_list[idx]]+=1
                    maxLanechange=self.getMaxLanechange(self.agent_list[idx])
                    relu=self.cum_lanechange_num[self.agent_list[idx]]-maxLanechange
                    neg_reward[idx]+= 0.5*np.tanh(relu/smooth_val) if relu>0 else 0
        #related to teleport penalty
        teleport_list=traci.simulation.getStartingTeleportIDList()
        for tp_veh in teleport_list:
            if tp_veh in self.agent_list:
                idx=self.agent_list.index(tp_veh)
                lane=traci.vehicle.getLaneID(tp_veh)
                if len(lane)==0: # error
                    neg_reward[idx]+=5.0
                else:
                    neg_reward[idx]+=traci.lane.getMaxSpeed(lane)/2.0
        
        # #related to emergency brake
        # emergency_stop_list = traci.simulation.getEmergencyStoppingVehiclesIDList()
        # for vehicle in emergency_stop_list:
        #     if vehicle in self.agent_list:
        #         idx = self.agent_list.index(vehicle)
        #         neg_reward[idx]+=10000
        
        return neg_reward

    #calculate overall reward
    def calculate_reward(self,removed_agent_idx):
        pos_reward = self.collect_pos_reward(removed_agent_idx)
        neg_reward = self.collect_neg_reward(removed_agent_idx)
        # print(list(pos_reward),list(neg_reward))
        if (pos_reward is None) and (neg_reward is None):
            reward = np.array([0.0],dtype=np.float32)
        else:
            self.reward += (pos_reward.sum()-neg_reward.sum())
            reward = pos_reward - neg_reward

        return reward


    """observation 관련 함수"""
    def get_observ_list(self):
        #observ = list()
        observ_list = [
            self.getSpeed,  #current speed of agent
            self.changeLaneRight,  #whether agent can make lane change to right
            self.changeLaneLeft,  #whether agent can make lane change to left
            self.getLeader,  #distance between leading car
            self.getFollower,  #distance between following car
            self.getLaneIndex, # index of current lane
            self.getRouteIndex, #index of current edge
            self.getDirection,  #direction of agent
            self.getRestDistance, # get the current position of total distance
            self.getTrafficLight, #traffic light status of current edge '''removed due to change of format of return'''
            self.getDesiredVelocity,
            self.getLanechangeNum,
        ]
        return observ_list
    
    def getLanechangeNum(self,agent):
        eps=1e-7
        maxLanechange=self.getMaxLanechange(agent)
        lanechangeRate=max(maxLanechange-self.cum_lanechange_num[agent],0)/(maxLanechange+eps)
        return lanechangeRate

    def getRestDistance(self,agent):
        lane_id=traci.vehicle.getLaneID(agent)
        if len(str(lane_id))>0:
            if lane_id[0]==':':# 교차로
                distance=1.0
            else:
                distance=traci.vehicle.getLanePosition(agent)
                #if agent == 'agent_0':
                #    print("distance:", distance)
                lane_length=traci.lane.getLength(lane_id)
                distance/=lane_length
                distance=1-distance
        else: # 위치를 못잡는 경우(start)
            distance=1.0
        return distance

    #return velocity of vehicle
    def getSpeed(self,agent):
        velocity=traci.vehicle.getSpeed(agent)
        return max(velocity,0.0)

    #check if agent can change to right lane
    def changeLaneRight(self, agent):
        changeLaneInfo = traci.vehicle.couldChangeLane(agent, -1)
        return changeLaneInfo

    #check if agent can change to left lane
    def changeLaneLeft(self, agent):
        changeLaneInfo = traci.vehicle.couldChangeLane(agent, 1)
        return changeLaneInfo

    #return distance from leading car, -1 if none
    def getLeader(self, agent):
        try:
            leadDistance = min(traci.vehicle.getLeader(agent, 0.0)[1],100)
        except TypeError:
            leadDistance = 100.0
        return leadDistance/100.0

    #return distance from following car, -1 if none
    def getFollower(self, agent):
        try:
            followDistance = min(traci.vehicle.getFollower(agent, 0.0)[1],100)
        except TypeError:
            followDistance = 100.0
        return followDistance/100.0

    #return index of current lane
    def getLaneIndex(self,agent):
        lane_idx=traci.vehicle.getLaneIndex(agent)
        lane_idx=max(lane_idx,0)
        lane_idx=min(lane_idx,self.num_lane)
        return lane_idx

    #return index of current route
    def getRouteIndex(self,agent):
        route_idx=traci.vehicle.getRouteIndex(agent)
        route_idx=max(route_idx,-1)
        route_idx=min(route_idx,self.num_edge)
        return route_idx

    def getMaxLanechange(self,agent):
        lane_id=traci.vehicle.getLaneID(agent)
        if len(lane_id)==0:
            return 5
        lane_length=traci.lane.getLength(lane_id)
        return lane_length//100+1

    # direction은 [current edge]-[surrounding edge]-[probability]의 순으로 구성된 중첩 dictionary
    # junction_edges는 map의 모든 junction에 대해 [junction(node)_id]-[surrounding edge]의 순으로 구성된 dictionary

    def getDirection(self, agent):
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
                        
        return next_edge_val

    """getDirection을 위한 함수들"""
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
    
    
    def getTrafficLight(self, agent):
        cur_edge = self.get_cur_edge(agent)
        next_node = 'n_' + cur_edge.split('_to_')[1]
        if next_node in traci.trafficlight.getIDList():
            tl_state = self.mapping_tl(cur_edge, next_node)
        else:
            #tl_state = -1 #not exist
            tl_state = [1.0,1.0,1.0,1.0,1.0]
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
            tl_state = [0.,0.,0.,0.,0.]
        elif tl_dict[cur_edge] == 'yyyyy': #대기
            tl_state = [0.2,0.2,0.2,0.2,0.2]
        elif tl_dict[cur_edge] == 'grrrg': #좌/우회전
            tl_state = [1.,0.,0.,0.,1.]
        elif tl_dict[cur_edge] == 'ggggr': #직진
            tl_state = [1.,1.,1.,1.,0.]
        return tl_state 
        
    def get_edge_from_edg_xml(self):
        add_file_path = os.path.join(
            self.file_path, 'Net_data', self.file_name + '.edg.xml')
        net_tree = parse(add_file_path)
        
        edges = net_tree.findall('edge')

        num_lanes = list()
        for edge in edges:
            num_lanes.append(int(edge.attrib['numLanes']))
        
        num_lane = max(num_lanes)-1 #lane index는 0부터 시작됨
        num_edge = len(edges)-1 #edge index는 0부터 시작됨

        return num_edge, num_lane
    