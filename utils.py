import os, json


def update_tensorBoard(writer, agent, env, epoch,configs):  # 메인에서 끌어오는 형식으로 해보려고 한다.
    #agent.update_tensorBoard   Loss, Learning Rate, Epsilon dqn으로 설정해놓음
    agent.update_tensorBoard(writer, epoch)
    #env.update_tensorBoard   Reward
    writer.add_scalar('episode/reward', env.reward.sum(),
                          configs['EXP_CONFIGS']['max_steps'] * epoch)

def save_params(agent, time_data):
        with open(os.path.join(agent.configs['current_path'], 'training_data', '{}.json'.format(time_data)), 'w') as fp:
            json.dump(agent.configs, fp, indent=2)

def load_params(agent, file_name):
    with open(os.path.join(agent.configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs
