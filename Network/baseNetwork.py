# import xml.etree.cElementTree as ET
from xml.etree.ElementTree import dump
from lxml import etree as ET
import os
E = ET.Element


class mainNetwork():
    def __init__(self, file_path, configs):
        if configs['network'] == "cross":
            from Network.cross import CrossNetwork
            self.network = CrossNetwork(file_path, configs['network'], configs)
        elif configs['network'] == "grid":
            from Network.grid import GridNetwork
            self.network = GridNetwork(file_path, configs['network'], configs)


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


class BaseNetwork():
    def __init__(self, file_path, file_name, configs):
        self.exp_configs = configs['EXP_CONFIGS']
        self.net_configs = configs['NET_CONFIGS']
        self.sim_start = 0
        self.max_steps = self.exp_configs['max_steps']
        self.current_path = file_path
        self.file_name = file_name
        self.current_Env_path = os.path.join(
            self.current_path, 'Net_data')
        if os.path.exists(self.current_Env_path) == False:
            os.mkdir(self.current_Env_path)
        if os.path.exists(self.current_Env_path) == False:
            os.mkdir(self.current_Env_path)

        self.num_cars = str(self.net_configs['num_cars'])
        self.lane_num = str(self.net_configs['numLanes'])
        self.flow_start = str(self.net_configs['flow_start'])
        self.flow_end = str(self.net_configs['flow_end'])
        self.laneLength = self.net_configs['laneLength']
        self.nodes = list()
        self.flows = list()
        self.vehicles = list()
        self.edges = list()
        self.connections = list()
        self.outputData = list()
        self.traffic_light = list()

    def specify_edge(self):
        edges = list()
        '''
        상속을 위한 함수
        '''
        return edges

    def specify_node(self):
        nodes = list()
        '''
        상속을 위한 함수
        '''

        return nodes

    def specify_flow(self):
        flows = list()
        '''
        상속을 위한 함수
        '''

        return flows
    
    def specify_outdata(self):
        outputData = list()
        '''
        상속을 위한 함수
        '''
        return outputData

    def specify_traffic_light(self):
        traffic_light = list()
        '''
        상속을 위한 함수
        '''
        return traffic_light

    #define vehicle type for agent 추가 210328
    def specify_vType(self):
        vType = list()
        vType.append({
            'id': 'rl_agent',
            'vClass': 'passenger',
            'length': '5',
            'accel': '3.5',
            'decel': '2.2',
            'sigma': '1.0', #sigma: driver's imperfection
            'maxSpeed': '10',
            'guiShape': 'passenger',
            'color': '1,0,0', #red
            'lcKeepRight': '0',
        })
        return vType

    def _generate_nod_xml(self):
        self.nodes = self.specify_node()
        nod_xml = ET.Element('nodes')

        for node_dict in self.nodes:
            nod_xml.append(E('node', attrib=node_dict))
            indent(nod_xml, 1)
        dump(nod_xml)
        tree = ET.ElementTree(nod_xml)
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
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.edg.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_net_xml(self):
        file_name_str = os.path.join(self.current_Env_path, self.file_name)
        
        if len(self.traffic_light) != 0:
            os.system('netconvert -n {0}.nod.xml -e {0}.edg.xml -i {0}_tl.add.xml -o {0}.net.xml --no-turnarounds 1'.format(
                file_name_str))
        elif len(self.connections) == 0:
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -o {}.net.xml --no-turnarounds 1'.format(
                file_name_str, file_name_str, file_name_str))
        else:  # connection이 존재하는 경우 -x
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -x {}.con.xml -o {}.net.xml --no-turnarounds 1'.format(
                file_name_str, file_name_str, file_name_str, file_name_str))

    def _generate_rou_xml(self):
        self.flows = self.specify_flow()
        self.routes = self.specify_route()
        self.vTypes = self.specify_vType()
        route_xml = ET.Element('routes')
        if len(self.vTypes) != 0:
            for _, vType_dict in enumerate(self.vTypes):
                route_xml.append(E('vType', attrib=vType_dict))
                indent(route_xml, 1)        
        if len(self.routes) != 0:
            for _, route_dict in enumerate(self.routes):
                route_xml.append(E('route', attrib=route_dict))
                indent(route_xml, 1)
        if len(self.flows) != 0:
            for _, flow_dict in enumerate(self.flows):
                route_xml.append(E('flow', attrib=flow_dict))
                indent(route_xml, 1)

        dump(route_xml)
        tree = ET.ElementTree(route_xml)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.rou.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_add_xml(self):
        traffic_light_set = self.specify_traffic_light()
        self.traffic_light = traffic_light_set

        tl_additional = ET.Element('additional')
        if len(self.traffic_light) != 0:
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
    
    def generate_cfg(self, route_exist, mode='simulate'):
        '''
        if all the generation over, inherit this function by `super`.
        '''
        sumocfg = ET.Element('configuration')
        inputXML = ET.SubElement(sumocfg, 'input')
        inputXML.append(
            E('net-file', attrib={'value': os.path.join(self.current_Env_path, self.file_name+'.net.xml')}))
        indent(sumocfg)
        if route_exist == True:
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
        dump(sumocfg)
        tree = ET.ElementTree(sumocfg)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.sumocfg'),
                   pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def test_net(self):
        self.generate_cfg(False)

        os.system('sumo-gui -c {}.sumocfg'.format(os.path.join(self.current_Env_path,
                                                               self.file_name+'_simulate')))

    def sumo_gui(self):
        self.generate_cfg(True)
        os.system('sumo-gui -c {}.sumocfg'.format(
            os.path.join(self.current_Env_path, self.file_name+'_simulate')))

    def generate_all_xml(self):
        self._generate_nod_xml()
        self._generate_edg_xml()
        self._generate_add_xml()
        self._generate_net_xml()
        self._generate_rou_xml()


if __name__ == '__main__':
    from configs import DEFAULT_CONFIGS
    network = BaseNetwork(DEFAULT_CONFIGS)
    network.generate_all_xml()
    network.generate_cfg(True)
