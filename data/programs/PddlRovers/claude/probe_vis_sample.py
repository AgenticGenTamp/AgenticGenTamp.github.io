import numpy as np
from probe_vis_lib import *

env,obs,info=new_env(1)
S=feats(obs,'sample3'); sp=np.array([S['x'],S['y']])
u=np.array([1.0,-1.0])/np.sqrt(2)  # approach from SE
def dist(obs,i=0): return np.linalg.norm(pose(obs,i)[:2]-sp)
def trysample(obs,i=0,th=None):
    o0=rf(obs,i)
    obs,_,_,_,_=st(env,op='sample',i=i)
    o1=rf(obs,i)
    s=feats(obs,'sample3')
    return obs, o1['store_full']>0.5 and o0['store_full']<0.5, o1, s

print("sample3",S)
# coarse: place at d then try
res=[]
for d in [0.35,0.30,0.28,0.26,0.25,0.24,0.20]:
    tgt=sp+u*d
    obs,ok=goto(env,obs,tgt[0],tgt[1],i=0)
    dd=dist(obs,0)
    obs,succ,r0,s3=trysample(obs,0)
    res.append((round(dd,4),succ,r0['store_full'],s3['analyzed_rover0'],r0['calibrated']))
    print("d=%.4f reach_ok=%s succ=%s store=%s an0=%s"%(dd,ok,succ,r0['store_full'],s3['analyzed_rover0']))
    if succ:
        obs,_,_,_,_=st(env,op='drop',i=0)
        print("   after drop store=",rf(obs,0)['store_full'])
print("steps used approx", )
env.close()
