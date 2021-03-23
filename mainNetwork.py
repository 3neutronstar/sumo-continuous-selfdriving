#GridNetwork 클래스 객체와 net_config를 튜플로 리턴

#network = NETWORK[flags.network]
#network = mainNetwork(configs).network

class mainNetwork():
    def __init__(self, flags):
        if flags.network == "net_1": #cross road 생성코드를 'net_1'로 수정할것
            from Network.net_1 import GridNetwork, NET_CONFIG 
            self.network = GridNetwork()
            network = self.network
            net_config = NET_CONFIG 
            return network, net_config        
        elif flags.network == "net_2": #grid.py를 'net_2.py'로 수정할것
            from Network.net_2 import GridNetwork 
            self.network = GridNetwork()
            network = self.network
            net_config = NET_CONFIG
            return network, net_config
        #elif:
        #elif:
        #else:

#추가할 NET_CONFIGS
#    'NET_CONFIGS': {
#        'grid_num',
#        'numLanes', 
#        'laneLength',
#        'flow_start',
#        'flow_end' 
#    }
