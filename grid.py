from RL_configs import EXP_CONFIGS
import torch
import math
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import dump
from lxml import etree as ET
import os
E = ET.Element

#Configuration Setting
Edges = list()
Nodes = list()
Vehicles = list()
EXP_CONFIGS = {
    'lane_num': 3,
    'model': 'normal',
    'file_name': '1x1grid',
    'laneLength': 300.0,
    'num_cars': 1800,
    'flow_start': 0,
    'flow_end': 3600,
    'sim_start': 0,
    'max_steps': 3600,
    'num_epochs': 1000,
    'edge_info': Edges,
    'node_info': Nodes,
    'vehicle_info': Vehicles,
}

#Indent function for file directory address set
def indent(elem, level=0):
    i = "\n  " + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + ""
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

class GridNetwork():
    def __init__(self, configs):
        self.configs = configs
        self.tl_rl_list = list()
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        
        self.sim_start = self.configs['sim_start']
        self.max_steps = self.configs['max_steps']
        self.current_path = os.path.dirname(os.path.abspath(__file__))

        self.file_name = self.configs['file_name']
        self.current_Env_path = os.path.join(
            self.current_path, 'Net_data')
        
        if os.path.exists(self.current_Env_path) == False:
            os.mkdir(self.current_Env_path)

        self.num_cars = str(self.configs['num_cars'])
        self.lane_num = str(self.configs['lane_num'])
        self.flow_start = str(self.configs['flow_start'])
        self.flow_end = str(self.configs['flow_end'])
        self.laneLength = self.configs['laneLength']
        self.nodes = list()
        self.flows = list()
        self.vehicles = list()
        self.edges = list()
        self.connections = list()
        self.outputData = list()
        self.traffic_light = list()

#Define network shape
#####################################################################################################
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

    def specify_flow(self):
        flows = list()
        direction_list = ['L', 'U', 'D', 'R']

        for _, edge in enumerate(self.edges): #전 edge 탐색
            for i, _ in enumerate(direction_list):
                if direction_list[i] in edge['from']: #edge의 outNode가 포함되는지 확인
                    for _, destEdge in enumerate(self.edges):
                        if destEdge['to'][-1] == direction_list[3-i]:
                            if destEdge['to'][-1] == direction_list[1] or destEdge['to'][-1] == direction_list[2]:
                                self.configs['vehsPerHour']='900' #수직 통행량 900
                            else:
                                self.configs['vehsPerHour']='1200' #수평 통행량 2000
                            
                            flows.append({
                                'from': edge['id'],
                                'to': destEdge['id'],
                                'id': edge['from'],
                                'begin': str(self.configs['flow_start']),
                                'end': str(self.configs['flow_end']),
                                'vehsPerHour':self.configs['vehsPerHour'],
                                'reroute': 'false',
                                'via': edge['id']+ ' ' +destEdge['id'],
                                'departPos': "base",
                                'departLane': 'best',
                            })
        self.flows = flows
        self.configs['vehicle_info'] = flows
        return flows

    def specify_connection(self):
        connections = list()

        self.connections = connections
        return connections

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

#Create network file
########################################################################################################
    def _generate_nod_xml(self):
        self.nodes = self.specify_node()
        nod_xml = ET.Element('nodes')

        for node_dict in self.nodes:
            # node_dict['x']=format(node_dict['x'],'.1f')
            nod_xml.append(E('node', attrib=node_dict))
            indent(nod_xml, 1)
        dump(nod_xml)
        tree = ET.ElementTree(nod_xml)
        # tree.write(self.file_name+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.nod.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_edg_xml(self):
        self.edges = self.specify_edge()
        edg_xml = ET.Element('edges')
        for _, edge_dict in enumerate(self.edges):
            edg_xml.append(E('edge', attrib=edge_dict))
            indent(edg_xml, 1)
        dump(edg_xml)
        tree = ET.ElementTree(edg_xml)
        # tree.write(self.xml_edg_name+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.edg.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_net_xml(self):
        # file_name_str=os.path.join(self.current_Env_path,self.file_name)
        file_name_str = os.path.join(self.current_Env_path, self.file_name)
        if len(self.traffic_light) != 0:
            os.system('netconvert -n {0}.nod.xml -e {0}.edg.xml -i {0}_tl.add.xml -o {0}.net.xml'.format(
                file_name_str))
        elif len(self.connections) == 0:
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -o {}.net.xml'.format(
                file_name_str, file_name_str, file_name_str))
        else:  # connection이 존재하는 경우 -x
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -x {}.con.xml -o {}.net.xml'.format(
                file_name_str, file_name_str, file_name_str, file_name_str))

    def _generate_rou_xml(self):
        self.flows = self.specify_flow()
        route_xml = ET.Element('routes')
        if len(self.vehicles) != 0:  # empty
            for _, vehicle_dict in enumerate(self.vehicles):
                route_xml.append(E('veh', attrib=vehicle_dict))
                indent(route_xml, 1)
        if len(self.flows) != 0:
            for _, flow_dict in enumerate(self.flows):
                route_xml.append(E('flow', attrib=flow_dict))
                indent(route_xml, 1)
        dump(route_xml)
        tree = ET.ElementTree(route_xml)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.rou.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_con_xml(self):
        self.cons = self.specify_connection()
        con_xml = ET.Element('connections')
        if len(self.connections) != 0:  # empty
            for _, connection_dict in enumerate(self.connections):
                con_xml.append(E('connection', attrib=connection_dict))
                indent(con_xml, 1)

        dump(con_xml)
        tree = ET.ElementTree(con_xml)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.con.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_add_xml(self):
        traffic_light_set = self.specify_traffic_light()
        self.traffic_light = traffic_light_set
        data_additional = ET.Element('additional')
        # edgeData와 landData파일의 생성위치는 data
        data_additional.append(E('edgeData', attrib={'id': 'edgeData_00', 'file': '{}_edge.xml'.format(self.current_path+'/data/'+self.file_name), 'begin': '0', 'end': str(
            self.configs['max_steps']), 'freq': '1000'}))
        indent(data_additional, 1)
        data_additional.append(E('laneData', attrib={'id': 'laneData_00', 'file': '{}_lane.xml'.format(self.current_path+'/data/'+self.file_name), 'begin': '0', 'end': str(
            self.configs['max_steps']), 'freq': '1000'}))
        indent(data_additional, 1)
        dump(data_additional)
        tree = ET.ElementTree(data_additional)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'_data.add.xml'),
                   pretty_print=True, encoding='UTF-8', xml_declaration=True)

        tl_additional = ET.Element('additional')

        for _, tl in enumerate(traffic_light_set):
            phase_set = tl.pop('phase')
            tlLogic = ET.SubElement(tl_additional, 'tlLogic', attrib=tl)
            indent(tl_additional, 1)
            for _, phase in enumerate(phase_set):
                tlLogic.append(E('phase', attrib=phase))
                indent(tl_additional, 2)

        dump(tl_additional)
        tree = ET.ElementTree(tl_additional)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'_tl.add.xml'),
                   pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def generate_cfg(self, route_exist): #from gen_net
        '''
        if all the generation over, inherit this function by `super`.
        '''
        sumocfg = ET.Element('configuration')
        inputXML = ET.SubElement(sumocfg, 'input')
        inputXML.append(
            E('net-file', attrib={'value': os.path.join(self.current_Env_path, self.file_name+'.net.xml')}))
        indent(sumocfg)
        if route_exist == True:
            #if self.configs['network'] == 'grid':  # grid에서만 생성
            #    self._generate_rou_xml()
            self._generate_rou_xml()
            if os.path.exists(os.path.join(self.current_Env_path, self.file_name+'.rou.xml')):
                inputXML.append(
                    E('route-files', attrib={'value': os.path.join(self.current_Env_path, self.file_name+'.rou.xml')}))
                indent(sumocfg)

        if os.path.exists(os.path.join(self.current_Env_path, self.file_name+'_data.add.xml')):
            inputXML.append(
                E('additional-files', attrib={'value': os.path.join(self.current_Env_path, self.file_name+'_data.add.xml')}))
            indent(sumocfg)

        time = ET.SubElement(sumocfg, 'time')
        time.append(E('begin', attrib={'value': str(self.sim_start)}))
        indent(sumocfg)
        time.append(E('end', attrib={'value': str(self.max_steps)}))
        indent(sumocfg)
        outputXML = ET.SubElement(sumocfg, 'output')
        indent(sumocfg)
        dump(sumocfg)
        
        tree = ET.ElementTree(sumocfg)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'_simulate.sumocfg'),
            pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def generate_all(self):
        self._generate_nod_xml()
        self._generate_edg_xml()
        self._generate_add_xml()
        self._generate_net_xml()
        self._generate_rou_xml()
        self.generate_cfg(True)


if __name__ == "__main__":
    configs = EXP_CONFIGS
    configs['file_name'] = '1x1grid'
    a = GridNetwork(configs)
    a.generate_all()
    a.generate_cfg(True)
