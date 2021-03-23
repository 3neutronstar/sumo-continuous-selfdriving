Vehicles = list()

EXP_CONFIGS = {
    'num_cars': 1800,
        'vehicle_info': Vehicles,
        'file_name': "3x3grid",
        'network': None,
        'sim_start' : 0,
        'max_step': 3600,
        'current_path': 'abspath',
        'gui': False,
        'mode' : 'simulate',
        

    # NET_CONFIGS
    # CALL FROM the Network dir
    'NET_CONFIGS': {
        'grid_num': 3,
        'lane_num': 3,  # lane 개수
        'laneLength' : 300,
        'flow_start' : 0,
        'flow_end' : 3600
    },

    # ENV_CONFIGS4
    # CALL FROM the Env dir
    'ENV_CONFIGS': {

    },

    # AGENT_CONFIGS
    # CALL FROM the Agent dir
    'AGENT_CONFIGS': {
        'algorithm': None
    }
}
