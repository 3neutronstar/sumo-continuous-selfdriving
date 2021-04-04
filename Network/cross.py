from Network.baseNetwork import BaseNetwork
import math
import torch

NET_CONFIGS = {
    'numLanes': 3,
    'laneLength': 300,
    'flow_start': 0,
    'flow_end': 3600,
    'num_cars': 1800,
}


class CrossNetwork(BaseNetwork):
    def __init__(self, file_path, file_name, configs):
        self.net_configs = NET_CONFIGS
        self.exp_configs = configs['EXP_CONFIGS']
        super().__init__(file_path, file_name, self.configs)

    # define nodes
    def specify_node(self):
        nodes = list()

        # define center_node
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
            'y': str('%.1f' % (self.net_configs['laneLength']))
        },
            {
            'id': 'n_D',
            'x': str('%.1f' % center),
            'y': str('%.1f' % -(self.net_configs['laneLength']))
        },
            {
            'id': 'n_R',
            'x': str('%.1f' % (self.net_configs['laneLength'])),
            'y': str('%.1f' % center)
        },
            {
            'id': 'n_L',
            'x': str('%.1f' % -(self.net_configs['laneLength'])),
            'y': str('%.1f' % center)
        }]

        for _, node_info in enumerate(node_information):
            nodes.append(node_info)
        return nodes

    # define edges
    def specify_edge(self):
        edges = list()
        direction_list = ['U', 'D', 'R', 'L']

        for _ in (direction_list):
            edge_info = dict()
            edge_info = {
                'from': 'n_C',
                'id': 'C_to_{}'.format(_),
                'to': 'n_{}'.format(_),
                'numLanes': str(self.net_configs['numLanes'])
            }
            edges.append(edge_info)
            edge_info = {
                'from': 'n_{}'.format(_),
                'id': '{}_to_C'.format(_),
                'to': 'n_C',
                'numLanes': str(self.net_configs['numLanes'])
            }
            edges.append(edge_info)
        return edges

    # define traffic flow
    def specify_flow(self):
        flows = list()
        destEdgeID = ''
        direction_list = ['L', 'U', 'D', 'R']

        for _, edge in enumerate(self.edges):  # search all edges
            for i, _ in enumerate(direction_list):
                if direction_list[i] in edge['from']:  # find only outNodes
                    destEdgeID = 'C_to_'+direction_list[3-i]

                    if destEdgeID[-1] == direction_list[1] or destEdgeID[-1] == direction_list[2]:
                        vehsPerHours = '900'  # 수직 통행량 900
                    else:
                        vehsPerHours = '1200'  # 수평 통행량 2000

                    flows.append({
                        'from': edge['id'],
                        'to': destEdgeID,
                        'id': edge['from'],
                        'begin': str(self.net_configs['flow_start']),
                        'end': str(self.net_configs['flow_end']),
                        'vehsPerHour': vehsPerHours,
                        'reroute': 'false',
                        'via': edge['id'] + ' ' + destEdgeID,
                        'departPos': "base",
                        'departLane': 'best',
                    })
        return flows

    # define traffic light
    def specify_traffic_light(self):
        traffic_lights = []
        numLanes = self.net_configs['numLanes']
        g = 'G'
        r = 'r'
        phase_set = [
            {'duration': '37',  # 1
                'state': 'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(  # 위좌아래좌
                    g*numLanes, g, r*numLanes, r),
             },
            {'duration': '3',
                'state': 'y'*(12+4*numLanes),
             },
            {'duration': '37',  # 2
                'state': 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(  # 위직아래직
                    g*numLanes, g, r*numLanes, r),  # current
             },
            {'duration': '3',
                'state': 'y'*(12+4*numLanes),
             },
            {'duration': '37',  # 1
                'state': 'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format(  # 좌좌우좌
                    g*numLanes, g, r*numLanes, r),
             },
            {'duration': '3',
                'state': 'y'*(12+4*numLanes),
             },
            {'duration': '37',  # 1
                'state': 'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(  # 좌직우직
                    g*numLanes, g, r*numLanes, r),  # current
             },
            {'duration': '3',
                'state': 'y'*(12+4*numLanes),
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

    # specify default route for agent
    def specify_route(self):
        route = list()
        route.append({
            'id': 'route_0',
            'edges': 'L_to_C C_to_D',
        })
        return route

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
    configs['NET_CONFIGS'] = NET_CONFIGS
    a = GridNetwork(configs)
    a.generate_cfg(True)
