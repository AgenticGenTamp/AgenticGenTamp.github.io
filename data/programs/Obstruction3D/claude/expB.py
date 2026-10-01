import sys, numpy as np
from env_client import make_env
import robot
from s2 import *
seed=int(sys.argv[1]); name=sys.argv[2]
env=make_env(); obs,info=env.reset(seed=seed)
p=opos(obs,name); hz=half(obs,name)
print(name,"pos",np.round(p,4),"half",np.round(hz,4),flush=True)
res=[]
for h in np.arange(0.09,0.019,-0.005):
    obs,bl,err=go_world(env,obs,p+np.array([0,0,h]),robot.down_R(0.0),nmax=60)
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0
    obs,r,t,tr,inf=env.step(a); ri=rinfo(obs)
    g=ri['grasp_active']
    res.append((round(h,3),'BLK' if bl else '', 'G' if g>0.5 else '.',round(err,3)))
    if g>0.5:
        a[10]=1.0; obs,r,t,tr,inf=env.step(a)
        print("  after release obj",np.round(opos(obs,name),4),"grasp",rinfo(obs)['grasp_active'],flush=True)
print(res,flush=True)
