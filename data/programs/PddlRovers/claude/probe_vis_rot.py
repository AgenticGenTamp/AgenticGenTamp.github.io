import numpy as np
from probe_vis_lib import *
def setth(env,obs,th,i):
    for _ in range(30):
        d=(th-pose(obs,i)[2]+np.pi)%(2*np.pi)-np.pi
        if abs(d)<0.003: break
        obs,_,_,_,_=st(env,dth=float(np.clip(d,-0.4,0.4)),i=i)
    return obs
for SEED,PIL,i,d0 in [(39,'obstacle5',0,1.24),(20,'obstacle7',0,1.12)]:
    env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    q=np.array([L[PIL]['x'],L[PIL]['y']]); dqo=np.linalg.norm(q-O); u=(q-O)/dqo
    free,xs,res=build_grid(obs)
    P=O+u*d0
    obs,ok=nav(env,obs,P[0],P[1],i=i,tol=0.005,free=free,xs=xs,res=res)
    p=pose(obs,i)
    print("== seed%d pillar%s center-dead-on at %s (target %s) f=%.2f"%(SEED,PIL,np.round(p[:2],3),np.round(P,3),1-dqo/d0))
    res_=[]
    for th in np.arange(-np.pi,np.pi-0.01,np.pi/6):
        obs=setth(env,obs,float(th),i)
        obs,_,_,_,_=st(env,op='calibrate',i=i)
        v=rf(obs,i)['calibrated']>0.5
        if v: obs,_,_,_,_=st(env,op='image',i=i)
        res_.append((round(float(pose(obs,i)[2]),2),v))
        print("   theta=%+.2f vis=%s"%(pose(obs,i)[2],v))
    env.close()
