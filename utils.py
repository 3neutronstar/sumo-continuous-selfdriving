import os
import json


# 메인에서 끌어오는 형식으로 해보려고 한다.
def update_tensorBoard(writer, agent, env, epoch, configs):
    # agent.update_tensorBoard   Loss, Learning Rate, Epsilon dqn으로 설정해놓음
    agent.update_tensorBoard(writer, epoch)
    # env.update_tensorBoard   Reward
    writer.add_scalar('episode/reward', env.reward.sum(),
                      configs['EXP_CONFIGS']['max_steps'] * epoch)


def save_params(file_path, time_data, configs):
    with open(os.path.join(file_path, 'training_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)


def load_params(file_path, file_name):
    with open(os.path.join(file_path, 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs
