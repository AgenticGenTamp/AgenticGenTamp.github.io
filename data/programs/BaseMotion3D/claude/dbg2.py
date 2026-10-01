import numpy as np
from env_client import make_env
import approach as A
env=make_env()
def run(seed,sx,sy,verbose=True):
    o,i=env.reset(seed=seed)
    for k in range(80):
        a=np.zeros(11); a[0]=np.clip(sx-o[0],-0.4,0.4); a[1]=np.clip(sy-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o2,r,t,tr,_=env.step(a)
        if np.allclose(o2[:2],o[:2]): break
        o=o2
    ap=A.GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(o,i)
    print("start",np.round(o[:2],3),"target",np.round(o[19:21],3))
    for k in range(60):
        a=ap.get_action(o)
        o,r,t,tr,_=env.step(a)
        if verbose and k<40: print(" ",k,np.round(a[:2],3),np.round(o[:2],3),ap.mode,None if ap.path is None else len(ap.path),ap.bound,t)
        if t or tr: break
    print("term",t,"steps",k+1)
run(22,1.6,-3.5)
