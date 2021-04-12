import os, json
# def load_params(file_path):
#     with open(os.path.join(file_path,'training_data')

# def save_params(file_path):
#     pass


def update_tensorBoard(writer, agent, env, epoch,configs):  # 메인에서 끌어오는 형식으로 해보려고 한다.
    #agent.update_tensorBoard   Loss, Learning Rate, Epsilon dqn으로 설정해놓음

    #env.update_tensorBoard   Reward
    writer.add_scalar('episode/reward', env.reward.sum(),
                          configs['EXP_CONFIGS']['max_steps'] * epoch)