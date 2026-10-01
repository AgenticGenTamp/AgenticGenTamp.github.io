import numpy as np
import kin
RD = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
JN = ['pos_arm_joint%d'%i for i in range(1,8)]
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info):
        r=state.get_object_from_name('robot')
        self.qi=np.array([state.get(r,j) for j in JN]); self.t=0; self.qt=None
    def get_action(self, s):
        r=s.get_object_from_name('robot')
        b=np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y'),s.get(r,'pos_base_rot')])
        a=np.zeros(11); self.t+=1
        a[0:3]=np.clip(np.array([-0.14,0.0,0.0])-b,-.1,.1)
        if self.t==10:
            pa=kin.world_to_arm(np.array([0.3,0.0,0.62]),b); Ra=kin.rotz(-b[2])@RD
            self.qt,_=kin.ik_arm(pa,Ra,self.qi,iters=300,lam=0.05,tool=-0.224)
        if self.qt is not None:
            d=np.clip((self.qt-self.qi)/0.25,-.1,.1); a[3:10]=d; self.qi=self.qi+0.25*d
        a[10]=0.0 if self.t<90 else 1.0
        return a
