from Agent.baseAgent import BaseAgent
import torch

class CrossAgent(BaseAgent):
    def __init__(self,file_path,configs):
        super(BaseAgent,self).__init__(file_path,configs)
    
    def get_action(self, states):
        actions=list()
        for state in states:
            actions.append(self.model(state))
        actions=torch.cat(actions,device=self.device)
        return actions

    def _model(self,state):
        lane_action=self.dqn_model(state)
        accel_action=self.ddpg_model(state,lane_action.detach().clone())
        action=torch.cat([lane_action,accel_action],dim=1)
        
        return action

    def update(self):
