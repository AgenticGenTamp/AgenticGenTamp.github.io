import numpy as np
from probe_vis_lib import *
env=make_env()
obs,info=env.reset(seed=0, options={'object_count':1})
print("object_count=1 ->",info,[n for n in obs.get_object_names() if n.startswith('objective')])
env.close()
env,obs,info=new_env(0)
Q=feats(obs,'obstacle9'); q=np.array([Q['x'],Q['y']]); print("pillar9",q)
def pushdir(obs,i,ux,uy):
    s=0.2
    while s>=0.0005:
        p0=pose(obs,i)[:2].copy()
        obs,_,_,_,_=st(env,dx=ux*s,dy=uy*s,i=i)
        if np.linalg.norm(pose(obs,i)[:2]-p0)<1e-7: s/=2
    return obs,pose(obs,i)[:2].copy()
# rover0 from start (1,-1.75): go to (0.654,-1.0) then push north into pillar
obs,ok=goto(env,obs,0.654,-1.0,i=0,tol=0.01); print("ok",ok,pose(obs,0))
obs,p=pushdir(obs,0,0,1); print("push north stop y=%.4f gap=%.4f (theta=%.2f)"%(p[1],q[1]-p[1],pose(obs,0)[2]))
# rotate 45 deg and push again
obs,_=goto(env,obs,0.654,p[1]-0.3,i=0,tol=0.01)
for _ in range(2): obs,_,_,_,_=st(env,dth=0.4,i=0)
print("theta now",pose(obs,0)[2])
obs,p2=pushdir(obs,0,0,1); print("push north stop y=%.4f gap=%.4f theta=%.2f"%(p2[1],q[1]-p2[1],pose(obs,0)[2]))
# push east into pillar from west side
obs,_=goto(env,obs,0.654-0.5,q[1],i=0,tol=0.02)
obs,p3=pushdir(obs,0,1,0); print("push east stop x=%.4f gap=%.4f theta=%.2f"%(p3[0],q[0]-p3[0],pose(obs,0)[2]))
env.close()
