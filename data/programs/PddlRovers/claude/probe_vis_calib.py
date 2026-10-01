import numpy as np
from probe_vis_lib import *
env,obs,info=new_env(1)
O=feats(obs,'objective0'); op=np.array([O['x'],O['y']])
print("obj0",O, "count",info)
# rover0 start (1,-1.75) -> go to x=2.118,y=-2.0 then north
obs,ok=goto(env,obs,op[0],-2.0,i=0,tol=0.01); print("to south point",ok,pose(obs,0))
prev=None
for k in range(40):
    p=pose(obs,0); d=np.linalg.norm(p[:2]-op)
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    c=rf(obs,0)['calibrated']
    if c>0.5:
        print("FIRST calibrate success at dist2d=%.4f prev_fail_dist=%s"%(d,prev)); break
    prev=round(d,4)
    obs,_,_,_,_=st(env,dy=0.2,i=0)
else:
    print("never calibrated")
# persistence: move around, check calibrated
for k in range(5):
    obs,_,_,_,_=st(env,dx=-0.2,dy=-0.2,dth=0.4,i=0)
print("calibrated after 5 moves:",rf(obs,0)['calibrated'],"dist now",np.linalg.norm(pose(obs,0)[:2]-op))
env.close()
