from gen_net import Network
from configs import EXP_CONFIGS
import math
import torch


class GridNetwork(Network):
    def __init__(self, configs):
        self.configs = configs
        super().__init__(self.configs)

    #define nodes    
    def specify_node(self):
        nodes = list()

        #define center_node
        center = 0
        node_info = dict()        
        node_information = [{
            'id': 'n_C',
            'type': 'traffic_light',
            'tl': 'n_C',
            'x': str('%.1f' % center),
            'y': str('%.1f' % center)
        },
            {
            'id': 'n_U',
            'x': str('%.1f' % center),
            'y': str('%.1f' % (self.configs['laneLength']))
        },
            {
            'id': 'n_D',
            'x': str('%.1f' % center),
            'y': str('%.1f' % -(self.configs['laneLength']))
        },
            {
            'id': 'n_R',
            'x': str('%.1f' % (self.configs['laneLength'])),
            'y': str('%.1f' % center)
        },
            {
            'id': 'n_L',
            'x': str('%.1f' % -(self.configs['laneLength'])),
            'y': str('%.1f' % center)
        }]
        
        for _, node_info in enumerate(node_information):
                nodes.append(node_info)
        self.configs['node_info'] = nodes
        self.nodes = nodes
        return nodes

    #define edges      
    def specify_edge(self):
        edges = list()
        direction_list = ['U', 'D', 'R', 'L']


        for _ in (direction_list):
            edge_info = dict()
            edge_info = {
                'from': 'n_C',
                'id': 'C_to_{}'.format(_),
                'to': 'n_{}'.format(_),
                'numLanes': self.lane_num
            }
            edges.append(edge_info)           
            edge_info = {
                'from': 'n_{}'.format(_),
                'id': '{}_to_C'.format(_),
                'to': 'n_C',
                'numLanes': self.lane_num
            }
            edges.append(edge_info)
        self.edges = edges
        self.configs['edge_info'] = edges
        return edges

    #define traffic flow
    def specify_flow(self):
        flows = list()
        destEdgeID = ''
        direction_list = ['L', 'U', 'D', 'R']
        
        for _, edge in enumerate(self.edges): #search all edges
            for i, _ in enumerate(direction_list):
                if direction_list[i] in edge['from']: #find only outNodes 
                    destEdgeID = 'n_'+direction_list[3-i]
                    
                    if destEdgeID[-1] == direction_list[1] or destEdgeID[-1] == direction_list[2]:
                        self.configs['vehsPerHour']='900' #수직 통행량 900
                    else:
                        self.configs['vehsPerHour']='1200' #수평 통행량 2000
                    
                    flows.append({
                    'from': edge['id'],
                    'to': destEdgeID,
                    'id': edge['from'],
                    'begin': str(self.configs['flow_start']),
                    'end': str(self.configs['flow_end']),
                    'vehsPerHour':self.configs['vehsPerHour'],
                    'reroute': 'false',
                    'via': edge['id']+ ' ' +destEdgeID,
                    'departPos': "base",
                    'departLane': 'best',
                })
        self.flows = flows
        self.configs['vehicle_info'] = flows
        return flows

    #define connections
    def specify_connection(self):
        connections = list()

        self.connections = connections
        return connections
    
    #define traffic light
    def specify_traffic_light(self):
        traffic_lights = []
        lane_num = self.configs['lane_num']
        g = 'G'
        r = 'r'
        phase_set = [
            {'duration': '37',  # 1
                'state': 'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(  # 위좌아래좌
                    g*lane_num, g, r*lane_num, r),
                },
            {'duration': '3',
                'state': 'y'*(12+4*lane_num),
                },
            {'duration': '37',  # 2
                'state': 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(  # 위직아래직
                    g*lane_num, g, r*lane_num, r),  # current
                },
            {'duration': '3',
                'state': 'y'*(12+4*lane_num),
                },
            {'duration': '37',  # 1
                'state': 'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format(  # 좌좌우좌
                    g*lane_num, g, r*lane_num, r),
                },
            {'duration': '3',
                'state': 'y'*(12+4*lane_num),
                },
            {'duration': '37',  # 1
                'state': 'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(  # 좌직우직
                    g*lane_num, g, r*lane_num, r),  # current
                },
            {'duration': '3',
                'state': 'y'*(12+4*lane_num),
                },
        ]

        traffic_lights.append({
            'id': 'n_C',
            'type': 'static',
            'programID': 'n_C',
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
    configs = EXP_CONFIGS
    configs['file_name'] = 'cross_road'
    a = GridNetwork(configs)
    a.generate_cfg(True)