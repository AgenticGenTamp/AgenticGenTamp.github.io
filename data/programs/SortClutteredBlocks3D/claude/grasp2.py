import numpy as np, sys, time
from env_client import make_env
from arm import *
def trial(mount_xy=(0.0,0.0), gz=0.024, R0=0.45, travel=0.13, yaw=0.0, seed=0, log=False):
    mount=np.array([mount_xy[0],mount_xy[1],0.386])
    env=make_env(); obs,info=env.reset(seed=seed)
    b,q,g=robot_state(obs)
    cb=cube_dict(obs); names=list(cb)
    tname=max(names, key=lambda n: min(np.linalg.norm(cb[n][:2]-cb[m][:2]) for m in names if m!=n))
    tgt=cb[tname]
    hist=[]
    def phase(z, grip, n, stop=None):
        nonlocal b,q,g,obs
        qt=ik(np.array([R0,0.0,z]), tool_R(yaw), q)
        for t in range(n):
            bxy=base_for(tgt[:2], q, mount); bt=np.array([bxy[0],bxy[1],TH])
            a=make_action(b,q,bt,qt,grip); obs,rw,te,tr,i=env.step(a); b,q,g=robot_state(obs)
            gw=gripper_world(b,q,mount)
            hist.append((round(float(gw[0]),4),round(float(gw[1]),4),round(float(gw[2]),4)))
            if stop and stop(q,qt): break
    conv=lambda q,qt: np.max(np.abs((qt-q+np.pi)%(2*np.pi)-np.pi))<0.01
    phase(travel, 0.0, 200, conv)
    n1=len(hist)
    phase(gz, 0.0, 80, conv)
    n2=len(hist)
    phase(gz, 1.0, 3)
    phase(travel, 1.0, 80, conv)
    cb2=cube_dict(obs)
    env.close()
    return tname, tgt, hist[n1-1], hist[n2-1], hist[-1], {k:np.round(cb2[k]-cb[k],3) for k in cb}, (n1,n2,len(hist))
t0=time.time()
print(trial())
print('time',time.time()-t0)
