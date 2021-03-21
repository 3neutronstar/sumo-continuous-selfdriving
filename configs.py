Vehicles = list()

EXP_CONFIGS = {
    'num_cars': 1800,
    'vehicle_info': Vehicles,
    'file_name': None,
    'network': None,
    'max_step': 3600,
    'current_path': 'abspath',
    'gui': False,

    # NET_CONFIGS
    # CALL FROM the Network dir
    'NET_CONFIGS': {
        'grid_num': 0,
        'lane_num': 0,  # lane 개수
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
