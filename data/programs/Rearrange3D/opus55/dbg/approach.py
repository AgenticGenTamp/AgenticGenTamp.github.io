import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
import kin
def Rdown(psi):
    c,s=np.cos(psi),np.sin(psi)
    return np.array([[c,s,0],[s,-c,0],[0,0,-1.]])
class GeneratedApproach:
    def __init__(self,action_space=None,observation_space=None,primitives=None): pass
    def reset(self,obs,info):
        self.QT=obs[96:103].astype(float).copy()
        base=obs[93:96]; cp=obs[32:35]
        Rt=kin.rotz(-base[2])@Rdown(0)
        ta=lambda pw: kin.rotz(-base[2])@(np.array(pw)-np.array([base[0],base[1],0]))-kin.MOUNT
        pre,_=kin.ik_arm(ta(cp+[0,0,0.15]),Rt,self.QT,iters=300)
        g,_=kin.ik_arm(ta(cp+[0,0,0.05]),Rt,pre)
        self.wps=[pre,g]; self.i=0; self.t=0
    def get_action(self,obs):
        q_t=self.wps[self.i]
        a=np.zeros(11,np.float32); a[3:10]=np.clip(4*(q_t-self.QT),-0.1,0.1)
        self.QT+=0.25*a[3:10]
        self.t+=1
        if np.max(np.abs(q_t-obs[96:103]))<0.003 and self.i<len(self.wps)-1: self.i+=1
        return a
