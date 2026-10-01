import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=1, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
free,xs,res=build_grid(obs)
u=np.array([-0.2911,-0.9567]); u/=np.linalg.norm(u)
print("O",np.round(O,4),"bearing_of_u",np.arctan2(u[1],u[0]))
def setth(obs,th,i=0):
    for _ in range(40):
        d=(th-pose(obs,i)[2]+np.pi)%(2*np.pi)-np.pi
        if abs(d)<0.002: break
        obs,_,_,_,_=st(env,dth=float(np.clip(d,-0.4,0.4)),i=i)
    return obs
obs,ok=nav(env,obs,*(O+u*1.7),i=0,tol=0.005,free=free,xs=xs,res=res)
out=[]
for th in np.arange(-np.pi,np.pi-0.01,np.pi/4):
    obs=setth(obs,float(th))
    d=1.60; obs,_=goto(env,obs,*(O+u*d),i=0,tol=0.002)
    obs,v=vis_test(env,obs,0)
    if not v:
        # walk inward
        while d>0.8:
            d-=0.05; obs,_=goto(env,obs,*(O+u*d),i=0,tol=0.002)
            obs,v=vis_test(env,obs,0)
            if v: break
    lastv=d
    while d<2.4:
        d+=0.05; obs,_=goto(env,obs,*(O+u*d),i=0,tol=0.002)
        obs,v=vis_test(env,obs,0)
        if not v: break
        lastv=d
    # refine
    lo,hi=lastv,d
    for _ in range(4):
        m=(lo+hi)/2; obs,_=goto(env,obs,*(O+u*m),i=0,tol=0.002)
        obs,v=vis_test(env,obs,0)
        if v: lo=m
        else: hi=m
    print("theta=%+.3f threshold center-dist=%.4f"%(pose(obs,0)[2],(lo+hi)/2))
    out.append((float(pose(obs,0)[2]),(lo+hi)/2))
np.save('thr_theta.npy',np.array(out))
env.close()
