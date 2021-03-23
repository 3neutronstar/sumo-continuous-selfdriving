from Network.baseNetwork import BaseNetwork
from configs import EXP_CONFIGS
import math
import torch

NET_CONFIG = {
    'numLanes': 3,
    'laneLength': 300,
    'grid_num': 3,
    'flow_start': 0,
    'flow_end': 3600,
    'edge_info': Edges,
    'node_info': Nodes,
}


class GridNetwork(BaseNetwork):
    def __init__(self, configs):
        self.net_configs = configs['NET_CONFIGS']
        self.exp_configs = configs['EXP_CONFIGS']
        super().__init__(self.configs)

    def specify_node(self):
        nodes = list()
        center = float(self.grid_num)/2.0
        for x in range(self.grid_num):
            for y in range(self.grid_num):
                node_info = dict()
                node_info = {
                    'id': 'n_'+str(x)+'_'+str(y),
                    'type': 'traffic_light',
                    'tl': 'n_'+str(x)+'_'+str(y),
                }
                grid_x = self.configs['laneLength']*(x-center)
                grid_y = self.configs['laneLength']*(center-y)

                node_info['x'] = str('%.1f' % grid_x)
                node_info['y'] = str('%.1f' % grid_y)
                nodes.append(node_info)
                self.tl_rl_list.append(node_info)

        for i in range(self.grid_num):
            grid_y = (center-i)*self.configs['laneLength']
            grid_x = (i-center)*self.configs['laneLength']
            node_information = [{
                'id': 'n_'+str(i)+'_u',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (-center*self.configs['laneLength']+(self.grid_num+1)*self.configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_r',
                'x': str('%.1f' % (-center*self.configs['laneLength']+(self.grid_num)*self.configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            },
                {
                'id': 'n_'+str(i)+'_d',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (+center*self.configs['laneLength']-(self.grid_num)*self.configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_l',
                'x': str('%.1f' % (+center*self.configs['laneLength']-(self.grid_num+1)*self.configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            }]
            for _, node_info in enumerate(node_information):
                nodes.append(node_info)
        self.configs['node_info'] = nodes
        self.nodes = nodes
        return nodes


    def specify_edge(self):
        edges = list()
        edges_dict = dict()
        for i in range(self.grid_num):
            edges_dict['n_{}_l'.format(i)] = list()
            edges_dict['n_{}_r'.format(i)] = list()
            edges_dict['n_{}_u'.format(i)] = list()
            edges_dict['n_{}_d'.format(i)] = list()

        for y in range(self.grid_num):
            for x in range(self.grid_num):
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
                if y == self.grid_num-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_d'.format(x))
                    edges_dict['n_{}_d'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if x == self.grid_num-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_r'.format(y))
                    edges_dict['n_{}_r'.format(y)].append(
                        'n_{}_{}'.format(x, y))

                # inside edge making
                if x+1 < self.grid_num:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x+1, y))

                if y+1 < self.grid_num:
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
                    'numLanes': self.numLanes
                }
                edges.append(edge_info)
        self.edges = edges
        self.configs['edge_info'] = edges
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
                                self.configs['vehsPerHour']='900'
                            else:
                                self.configs['vehsPerHour']='2000'
                            via_string = str()
                            node_x_y = edge['id'][2]
                            if 'r' in edge['id']:
                                for i in range(self.configs['grid_num']-1, 0, -1):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        i, node_x_y, i-1, node_x_y)
                            elif 'l' in edge['id']:
                                for i in range(self.configs['grid_num']-2):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        i, node_x_y, i+1, node_x_y)
                            elif 'u' in edge['id']:
                                for i in range(self.configs['grid_num']-2):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        node_x_y, i, node_x_y, i+1)
                            elif 'd' in edge['id']:
                                for i in range(self.configs['grid_num']-1, 0, -1):
                                    via_string += 'n_{}_{}_to_n_{}_{} '.format(
                                        node_x_y, i, node_x_y, i-1)

                            flows.append({
                                'from': edge['id'],
                                'to': checkEdge['id'],
                                'id': edge['from'],
                                'begin': str(self.configs['flow_start']),
                                'end': str(self.configs['flow_end']),
                                'vehsPerHour':self.configs['vehsPerHour'],
                                'reroute': 'false',
                                'via': edge['id']+" "+via_string+" "+checkEdge['id'],
                                'departPos': "base",
                                'departLane': 'best',
                            })

        self.flows = flows
        self.configs['vehicle_info'] = flows
        return flows

    # define connections
    def specify_connection(self):
        connections = list()

        self.connections = connections
        return connections

    # define traffic light
    def specify_traffic_light(self):
        traffic_lights = []
        num_lanes = self.configs['num_lanes']
        g = 'G'
        r = 'r'
        for i in range(self.grid_num):
            for j in range(self.grid_num):
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

    def generate_cfg(self, route_exist, mode='simulate'):
        self._generate_nod_xml()
        self._generate_edg_xml()
        self._generate_add_xml()
        self._generate_net_xml()
        super().generate_cfg(route_exist, mode)


if __name__ == "__main__":
    from configs import EXP_CONFIGS
    configs = dict()
    configs['EXP_CONFIGS'] = EXP_CONFIGS
    configs['NET_CONFIGS'] = NET_CONFIG
    a = GridNetwork(configs)
    a.generate_cfg(True)
