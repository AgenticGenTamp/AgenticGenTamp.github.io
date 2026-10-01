import numpy as np, sys
from probe_vis_lib import *
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 39
R=float(sys.argv[2]) if len(sys.argv)>2 else 1.2
env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
print("seed",SEED,"O",np.round(O,3),"R",R)
free,xs,res=build_grid(obs)
def vis(obs,i):
    if rf(obs,i)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=i)
    obs,_,_,_,_=st(env,op='calibrate',i=i)
    return obs, rf(obs,i)['calibrated']>0.5
out=[]
angs=np.arange(-np.pi+0.05,0.0,0.15)
for th in angs:
    P=O+R*np.array([np.cos(th),np.sin(th)])
    if abs(P[0])<0.32 or abs(P[0])>2.05 or abs(P[1])>2.05: continue
    i=0 if P[0]>0 else 1
    obs,ok=nav(env,obs,P[0],P[1],i=i,tol=0.01,free=free,xs=xs,res=res)
    p=pose(obs,i)[:2]
    if np.linalg.norm(p-P)>0.05: 
        print(" bearing %+.2f UNREACHED (at %s)"%(th,np.round(p,2))); continue
    obs,v=vis(obs,i)
    b=np.arctan2(p[1]-O[1],p[0]-O[0]); d=np.linalg.norm(p-O)
    out.append((round(float(b),3),v,round(float(d),3)))
    print(" bearing %+.3f d=%.3f vis=%s"%(b,d,v))
env.close()
