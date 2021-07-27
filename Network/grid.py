from Network.baseNetwork import BaseNetwork
import math
import torch

NET_CONFIGS = {
    'numLanes': 3,
    'laneLength': 300,
    'grid_num': 3,
    'flow_start': 0,
    'flow_end': 3600,
    'num_cars': 1800,
}


class GridNetwork(BaseNetwork):
    def __init__(self, file_path, file_name, configs):
        configs['NET_CONFIGS'] = NET_CONFIGS
        self.net_configs = NET_CONFIGS
        self.exp_configs = configs['EXP_CONFIGS']
        super().__init__(file_path, file_name, configs)

    def specify_node(self):
        nodes = list()
        center = float(self.net_configs['grid_num'])/2.0
        for x in range(self.net_configs['grid_num']):
            for y in range(self.net_configs['grid_num']):
                node_info = dict()
                node_info = {
                    'id': 'n_'+str(x)+'_'+str(y),
                    'type': 'traffic_light',
                    'tl': 'n_'+str(x)+'_'+str(y),
                }
                grid_x = self.net_configs['laneLength']*(x-center)
                grid_y = self.net_configs['laneLength']*(center-y)

                node_info['x'] = str('%.1f' % grid_x)
                node_info['y'] = str('%.1f' % grid_y)
                nodes.append(node_info)
                # self.tl_rl_list.append(node_info)

        for i in range(self.net_configs['grid_num']):
            grid_y = (center-i)*self.net_configs['laneLength']
            grid_x = (i-center)*self.net_configs['laneLength']
            node_information = [{
                'id': 'n_'+str(i)+'_u',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (-center*self.net_configs['laneLength']+(self.net_configs['grid_num']+1)*self.net_configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_r',
                'x': str('%.1f' % (-center*self.net_configs['laneLength']+(self.net_configs['grid_num'])*self.net_configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            },
                {
                'id': 'n_'+str(i)+'_d',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (+center*self.net_configs['laneLength']-(self.net_configs['grid_num'])*self.net_configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_l',
                'x': str('%.1f' % (+center*self.net_configs['laneLength']-(self.net_configs['grid_num']+1)*self.net_configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            }]
            for _, node_info in enumerate(node_information):
                nodes.append(node_info)
        self.net_configs['node_info'] = nodes
        self.nodes = nodes
        return nodes

    def specify_edge(self):
        edges = list()
        edges_dict = dict()
        for i in range(self.net_configs['grid_num']):
            edges_dict['n_{}_l'.format(i)] = list()
            edges_dict['n_{}_r'.format(i)] = list()
            edges_dict['n_{}_u'.format(i)] = list()
            edges_dict['n_{}_d'.format(i)] = list()

        for y in range(self.net_configs['grid_num']):
            for x in range(self.net_configs['grid_num']):
                edges_dict['n_{}_{}'.format(x, y)] = list()

                # outside edge making
                if x == 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_l'.format(y))
                    edges_dict['n_{}_l'.format(y)].append(
                        'n_{}_{}'.format(x, y))
                if y == 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_u'.format(x))
                    edges_dict['n_{}_u'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if y == self.net_configs['grid_num']-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_d'.format(x))
                    edges_dict['n_{}_d'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if x == self.net_configs['grid_num']-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_r'.format(y))
                    edges_dict['n_{}_r'.format(y)].append(
                        'n_{}_{}'.format(x, y))

                # inside edge making
                if x+1 < self.net_configs['grid_num']:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x+1, y))

                if y+1 < self.net_configs['grid_num']:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x, y+1))
                if x-1 >= 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x-1, y))
                if y-1 >= 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x, y-1))

        for _, dict_key in enumerate(edges_dict.keys()):
            for i, _ in enumerate(edges_dict[dict_key]):
                edge_info = dict()
                edge_info = {
                    'from': dict_key,
                    'id': "{}_to_{}".format(dict_key, edges_dict[dict_key][i]),
                    'to': edges_dict[dict_key][i],
                    'numLanes': str(self.net_configs['numLanes'])
                }
                edges.append(edge_info)
        self.edges = edges
        self.net_configs['edge_info'] = edges
        return edges

    def specify_flow(self):
        flows = list()
        direction_list = ['l', 'u', 'd', 'r']

        for _, edge in enumerate(self.edges):
            for i, _ in enumerate(direction_list):
                if direction_list[i] in edge['from']:
                    for _, checkEdge in enumerate(self.edges):
                        if edge['from'][-3] == checkEdge['to'][-3] and checkEdge['to'][-1] == direction_list[3-i] and direction_list[i] in edge['from']:

                            if checkEdge['to'][-1] == direction_list[1] or checkEdge['to'][-1] == direction_list[2]:
                                self.net_configs['vehsPerHour'] = '900'
                            else:
                                self.net_configs['vehsPerHour'] = '2000'
                            via_string = str()
                            node_x_y = edge['id'][2]
                            if 'r' in edge['id']:
                                for i in range(self.net_configs['grid_num']-1, 0, -1):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        i, node_x_y, i-1, node_x_y)
                            elif 'l' in edge['id']:
                                for i in range(self.net_configs['grid_num']-2):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        i, node_x_y, i+1, node_x_y)
                            elif 'u' in edge['id']:
                                for i in range(self.net_configs['grid_num']-2):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        node_x_y, i, node_x_y, i+1)
                            elif 'd' in edge['id']:
                                for i in range(self.net_configs['grid_num']-1, 0, -1):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        node_x_y, i, node_x_y, i-1)

                            flows.append({
                                'from': edge['id'],
                                'to': checkEdge['id'],
                                'id': edge['from'],
                                'begin': str(self.net_configs['flow_start']),
                                'end': str(self.net_configs['flow_end']),
                                'vehsPerHour': self.net_configs['vehsPerHour'],
                                'reroute': 'false',
                                'via': edge['id']+" "+via_string+" "+checkEdge['id'],
                                'departPos': "base",
                                'departLane': 'best',
                            })

        self.flows = flows
        self.net_configs['vehicle_info'] = flows
        return flows

    # define connections
    def specify_connection(self):
        connections = list()

        self.connections = connections
        return connections

    # define traffic light
    def specify_traffic_light(self):
        traffic_lights = []
        num_lanes = self.net_configs['numLanes']
        g = 'G'
        r = 'r'
        for i in range(self.net_configs['grid_num']):
            for j in range(self.net_configs['grid_num']):
                phase_set = [
                    {'duration': '37',  # 1
                     'state': 'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(  # 위좌아래좌
                         g*num_lanes, g, r*num_lanes, r),
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                    {'duration': '37',  # 2
                     'state': 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(  # 위직아래직
                         g*num_lanes, g, r*num_lanes, r),  # current
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                    {'duration': '37',  # 1
                     'state': 'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format(  # 좌좌우좌
                         g*num_lanes, g, r*num_lanes, r),
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                    {'duration': '37',  # 1
                     'state': 'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(  # 좌직우직
                         g*num_lanes, g, r*num_lanes, r),  # current
                     },
                    {'duration': '3',
                     'state': 'y'*(12+4*num_lanes),
                     },
                ]

                traffic_lights.append({
                    'id': 'n_{}_{}'.format(i, j),
                    'type': 'static',
                    'programID': 'n_{}_{}'.format(i, j),
                    'offset': '0',
                    'phase': phase_set,
                })
        return traffic_lights


#재귀적 탐색 알고리즘
    def rec_search(self, before_edge, edge_list, route, routes, n):
        direction = ['u', 'r', 'd', 'l']
        edge_temp = list()

        for next_edge in edge_list:
            if next_edge[:5] == before_edge[-5:] and next_edge[-5:] != before_edge[:5]:
                edge_temp.append(next_edge) #이전 edge에 맞붙은 3개의 edge를 리스트화
                
                for edge in edge_temp:
                    if edge[-1] in direction:
                        routes.append({'id': 'route_'+str(n),
                            'edges': route + ' ' + edge})
                        n += 1
                    else:
                        route = route + ' ' + edge 
                        self.rec_search(edge, edge_list, route, routes, n)


    #define routes
    #edge example: n_0_l_to_n_0_0 n_0_0_to_n_0_1(총 13)
    def specify_route(self):
        routes = list()
        n = 0
        direction = ['u', 'r', 'd', 'l']
        edge_list_raw = self.specify_edge()
        edge_list = list()
        
        for edge_dic in self.specify_edge():
            edge_list.append(edge_dic['id'])

        for edge_0 in edge_list: 
            if edge_0[4] in direction:
                route = edge_0 #map을 빠져나가는 모든 edge에 대하여

                self.rec_search(edge_0, edge_list, route, routes, n)
        return routes


    def generate_cfg(self, route_exist, mode='simulate'):
        self._generate_nod_xml()
        self._generate_edg_xml()
        self._generate_add_xml()
        self._generate_net_xml()
        super().generate_cfg(route_exist, mode)


if __name__ == "__main__":
    from configs import DEFAULT_CONFIGS
    configs = DEFAULT_CONFIGS
    configs['NET_CONFIGS'] = NET_CONFIGS
    a = GridNetwork(configs)
    a.generate_cfg(True)
