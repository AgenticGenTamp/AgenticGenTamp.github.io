import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=1, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
print("O",np.round(O,4))
free,xs,res=build_grid(obs)
def setth(obs,th,i=0):
    for _ in range(40):
        d=th-pose(obs,i)[2]
        d=(d+np.pi)%(2*np.pi)-np.pi
        if abs(d)<0.002: break
        obs,_,_,_,_=st(env,dth=float(np.clip(d,-0.4,0.4)),i=i)
    return obs
u=np.array([-0.2911,-0.9567]) # bearing used before (south-southwest of O)
u=u/np.linalg.norm(u)
obs,ok=nav(env,obs,*(O+u*1.85),i=0,tol=0.005,free=free,xs=xs,res=res)
print("at",np.round(pose(obs,0),3),"d",np.linalg.norm(pose(obs,0)[:2]-O),ok)
for th in [np.pi, 0.0, np.pi/2, -np.pi/2, np.pi]:
    obs=setth(obs,th)
    # back to d=1.95 then step outward
    obs,_=goto(env,obs,*(O+u*1.95),i=0,tol=0.002)
    d=1.95; found=None
    for k in range(40):
        obs,v=vis_test(env,obs,0)
        if not v: found=d; break
        d+=0.01
        obs,_=goto(env,obs,*(O+u*d),i=0,tol=0.002)
    print("theta=%+.3f  first invisible center-dist=%s"%(pose(obs,0)[2],found))
env.close()
