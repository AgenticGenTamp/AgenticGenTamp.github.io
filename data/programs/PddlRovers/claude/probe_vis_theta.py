import numpy as np
from probe_vis_lib import *
env,obs,info=new_env(1)
O=feats(obs,'objective0'); op=np.array([O['x'],O['y']])
start=pose(obs,0)[:2]; u=(start-op); u/=np.linalg.norm(u)
tgt=op+u*1.0
obs,ok=goto(env,obs,tgt[0],tgt[1],i=0,tol=0.005)
p=pose(obs,0); print("pos",p,"d",np.linalg.norm(p[:2]-op),"bearing",np.arctan2(op[1]-p[1],op[0]-p[0]))
# sweep theta
for k in range(17):
    # clear calibrated
    if rf(obs,0)['calibrated']>0.5:
        obs,_,_,_,_=st(env,op='image',i=0)
    c0=rf(obs,0)['calibrated']
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    c1=rf(obs,0)['calibrated']
    print("theta=%+.3f calib %s->%s"%(pose(obs,0)[2],c0,c1))
    for _ in range(1): obs,_,_,_,_=st(env,dth=0.4,i=0)
print("obj feats", {n:feats(obs,n) for n in ['objective0','objective1']})
env.close()
