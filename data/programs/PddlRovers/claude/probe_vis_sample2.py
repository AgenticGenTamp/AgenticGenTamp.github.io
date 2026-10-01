import numpy as np
from probe_vis_lib import *
env,obs,info=new_env(1)
S=feats(obs,'sample3'); sp=np.array([S['x'],S['y']])
u=np.array([1.0,-1.0])/np.sqrt(2)
def place(obs,d,i=0,th=None):
    tgt=sp+u*d
    obs,ok=goto(env,obs,tgt[0],tgt[1],i=i,tol=0.0002,maxit=200)
    return obs,np.linalg.norm(pose(obs,i)[:2]-sp)
def trys(obs,i=0):
    b=rf(obs,i)['store_full']
    obs,_,_,_,_=st(env,op='sample',i=i)
    a=rf(obs,i)['store_full']
    return obs, (a>0.5 and b<0.5)
lo,hi=0.24,0.30
for it in range(9):
    mid=(lo+hi)/2
    obs,dd=place(obs,mid)
    obs,s=trys(obs)
    if s:
        obs,_,_,_,_=st(env,op='drop')
        lo=dd
    else: hi=dd
print("SAMPLE RADIUS in [%.5f, %.5f]"%(lo,hi))
# heading test at d=0.24 with various theta
obs,dd=place(obs,0.24)
for th in [0.4,1.2,2.0,3.0]:
    for _ in range(10):
        obs,_,_,_,_=st(env,dth=0.4)
    obs,s=trys(obs)
    print("theta=%.2f sample succ=%s"%(pose(obs)[2],s))
    if s: obs,_,_,_,_=st(env,op='drop')
# store_full blocking
obs,s=trys(obs); print("sample with empty store:",s)
obs,s2=trys(obs); print("sample again while full:",s2, "store",rf(obs)['store_full'])
obs,_,_,_,_=st(env,op='drop')
# rover1 samples same sample3
obs,ok=goto(env,obs,sp[0]-0.15,sp[1]-0.15,i=1,tol=0.01)
obs,s=trys(obs,1)
print("rover1 sample same sample3:",s,"an1",feats(obs,'sample3')['analyzed_rover1'],"reach",ok,pose(obs,1))
print("sample3 feats",feats(obs,'sample3'))
env.close()
