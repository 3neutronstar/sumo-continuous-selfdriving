import os, json

# def load_params(file_path):
#     with open(os.path.join(file_path,'training_data')

# def save_params(file_path):
#     pass


def update_tensorBoard(writer, agent, env, epoch):  # 메인에서 끌어오는 형식으로 해보려고 한다.
    #agent.update_tensorBoard   Loss, Learning Rate, Epsilon dqn으로 설정해놓음
    writer.add_scalar('loss', agent.dqn_model.running_loss / agent.configs['max_steps'],
                        agent.configs['max_steps'] * epoch)
    writer.add_scalar('learning_rate', agent.dqn_model.optimizer.param_groups[0]['lr'],
                        agent.configs['max_steps'] * epoch)
    writer.add_scalar('epsilon',
                        agent.dqn_model.epsilon, agent.configs['max_steps'] * epoch)

    #env.update_tensorBoard   Reward
    writer.add_scalar('episode/reward', env.reward.sum(),
                          env.configs['max_steps'] * epoch)