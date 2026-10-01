import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=action_space
    def reset(self,state,info): self.t=0
    def get_action(self,state):
        r=state.get_object_from_name("robot")
        b=np.array([state.get(r,f) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
        q=np.array([state.get(r,"pos_arm_joint%d"%i) for i in range(1,8)])
        c=state.get_object_from_name("cube3"); cy=state.get(c,"y")
        bg=np.array([1.,cy,np.pi]) if self.t<100 else np.array([.77,cy,np.pi])
        qg=np.array([0.,1.34,np.pi,-1.4,0.,1.,np.pi/2])
        if self.t>=180: qg[1]=.75
        a=np.zeros(11,np.float32);e=bg-b;e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi
        a[:3]=np.clip(e,-.1,.1);a[3:10]=np.clip(qg-q,-.1,.1)
        a[10]=1. if self.t<150 else 0.;self.t+=1;return a
