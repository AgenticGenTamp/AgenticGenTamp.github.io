"""Visual-only joint target probe (not the evaluated policy)."""
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
        self.targets=[np.array([-.146,2.24,np.pi,-1.20,0.0,q6,np.pi/2])
                      for q6 in (.4,0.0,-.5,-1.0,-1.5,-2.0)]
        self.step=0
    def reset(self,state,info): self.step=0
    def get_action(self,state):
        r=state.get_object_from_name("robot")
        q=np.array([state.get(r,"pos_arm_joint%d"%i) for i in range(1,8)])
        target=self.targets[min(self.step//60,len(self.targets)-1)]
        a=np.zeros(11,np.float32); a[3:10]=np.clip(1.8*(target-q),-.1,.1); a[10]=0
        self.step+=1
        return a
