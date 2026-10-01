import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
class E:
    def __init__(self, seed):
        self.env=make_env(); self.obs,_=self.env.reset(seed=seed); self.t=0; self.done=False
    def st(self,a):
        self.obs,r,te,tr,_=self.env.step(np.array(a,dtype=np.float32)); self.t+=1; self.done=te
        return te
    def goto(self,x,y,th,arm=None,vac=0,n=300,verbose=True):
        o=self.obs
        for i in range(n):
            o=self.obs
            dx=np.clip(x-o[0],-.05,.05); dy=np.clip(y-o[1],-.05,.05); dt=np.clip(wrap(th-o[2]),-.196,.196)
            da=0 if arm is None else np.clip(arm-o[4],-.1,.1)
            if max(abs(dx),abs(dy),abs(dt),abs(da))<1e-5: return True
            prev=o.copy(); self.st([dx,dy,dt,da,vac])
            if np.allclose(prev[:5],self.obs[:5]):
                if verbose: print('stuck',self.obs[:5],'goal',x,y,th)
                return False
        return False
    def grasp_hook(self, s=1.2):
        o=self.obs; hx,hy,ht=o[9:12]
        a=np.array([np.cos(ht),np.sin(ht)]); u=-a
        n=np.array([u[1],-u[0]])
        P=np.array([hx,hy])+u*s
        # choose side with lower y robot
        if n[1]>0: n=-n
        if P[1]+n[1]*0.35>1.12 or P[1]+n[1]*0.35<0.1: n=-n
        R=P+n*0.35; th=np.arctan2(-n[1],-n[0])
        self.goto(R[0],R[1],th,arm=0.1)
        for i in range(40):
            prev=self.obs.copy(); self.st([-n[0]*0.01,-n[1]*0.01,0,0,0])
            if np.allclose(prev[:2],self.obs[:2]): break
        self.st([0,0,0,0,1])
        o=self.obs
        c,s_=np.cos(o[2]),np.sin(o[2])
        d=o[9:11]-o[0:2]
        self.off=np.array([c*d[0]+s_*d[1], -s_*d[0]+c*d[1]]); self.dth=wrap(o[11]-o[2])
    def robot_for_hook(self,vx,vy,hth):
        rth=wrap(hth-self.dth); c,s=np.cos(rth),np.sin(rth)
        return vx-(c*self.off[0]-s*self.off[1]), vy-(s*self.off[0]+c*self.off[1]), rth
    def hook_goto(self,vx,vy,hth,n=300,verbose=True):
        x,y,th=self.robot_for_hook(vx,vy,hth)
        return self.goto(x,y,th,vac=1,n=n,verbose=verbose)
