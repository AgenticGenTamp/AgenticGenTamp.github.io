import numpy as np, sys
from probe_vis_lib import *
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 39
env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
print("seed",SEED,"O",np.round(O,4))
for n in sorted(L):
    if n.startswith('obstacle'):
        f=L[n]; print("  ",n,round(f['x'],3),round(f['y'],3),"half",f['half_x'],f['half_z'])
free,xs,res=build_grid(obs)
def vis(obs,i):
    if rf(obs,i)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=i)
    obs,_,_,_,_=st(env,op='calibrate',i=i)
    return obs, rf(obs,i)['calibrated']>0.5
def at(obs,P,i,first=False):
    if first: return nav(env,obs,P[0],P[1],i=i,tol=0.005,free=free,xs=xs,res=res)
    return goto(env,obs,P[0],P[1],i=i,tol=0.003)
for th in [-2.6,-2.3,-1.9,-1.6,-1.3,-1.0,-0.7,-0.45]:
    d=0.8; e=np.array([np.cos(th),np.sin(th)])
    P=O+d*e
    if abs(P[0])<0.32 or abs(P[0])>2.05 or abs(P[1])>2.05: continue
    i=0 if P[0]>0 else 1
    obs,ok=at(obs,P,i,True)
    obs,v=vis(obs,i)
    if not v:
        print("bearing %+.2f: not visible even at d=0.8 (pos %s)"%(th,np.round(pose(obs,i)[:2],2))); continue
    # step outward coarse 0.05 then refine 0.005
    last_ok=d
    fail=None
    while d<2.3:
        d+=0.05; P=O+d*e
        if abs(P[0])<0.32 or abs(P[0])>2.10 or abs(P[1])>2.10: fail='bounds'; break
        obs,ok2=at(obs,P,i)
        if np.linalg.norm(pose(obs,i)[:2]-P)>0.03: fail='blocked'; break
        obs,v=vis(obs,i)
        if not v: fail=d; break
        last_ok=d
    if isinstance(fail,float):
        lo,hi=last_ok,fail
        for _ in range(4):
            m=(lo+hi)/2; P=O+m*e
            obs,_=at(obs,P,i)
            obs,v=vis(obs,i)
            if v: lo=m
            else: hi=m
        print("bearing %+.2f  r_max in [%.3f, %.3f]"%(th,lo,hi))
    else:
        print("bearing %+.2f  visible up to d=%.2f (stopped: %s)"%(th,last_ok,fail))
env.close()
