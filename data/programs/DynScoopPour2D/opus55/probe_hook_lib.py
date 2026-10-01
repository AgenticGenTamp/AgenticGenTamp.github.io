import numpy as np
from env_client import make_env
RF=['x','y','theta','arm_joint','finger_gap']
HF=['x','y','theta','held']
class P:
    def __init__(s,seed=0):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed)
    def r(s): o=s.obs.get_object_from_name('robot'); return {f:round(float(s.obs.get(o,f)),4) for f in RF}
    def h(s): o=s.obs.get_object_from_name('hook'); return {f:round(float(s.obs.get(o,f)),4) for f in HF}
    def step(s,a,n=1):
        for _ in range(n):
            s.obs,rw,te,tr,info=s.env.step(np.array(a,dtype=float))
        return s
    def goto(s,x=None,y=None,th=None,arm=None,gap=None,maxn=300,tol=1e-3):
        for i in range(maxn):
            r=s.r(); a=[0]*5
            if x is not None: a[0]=np.clip(x-r['x'],-.03,.03)
            if y is not None: a[1]=np.clip(y-r['y'],-.03,.03)
            if th is not None: a[2]=np.clip((th-r['theta']+np.pi)%(2*np.pi)-np.pi,-.098,.098)
            if arm is not None: a[3]=np.clip(arm-r['arm_joint'],-.08,.08)
            if gap is not None: a[4]=np.clip(gap-r['finger_gap'],-.015,.015)
            if max(abs(v) for v in a)<tol: return i
            s.step(a)
        return maxn
