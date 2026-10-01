import numpy as np
from env_client import make_env
from arm import robot_state, make_action
from fk import fk, ik
import approach as A
# test1: base moves fast in y with arm fixed
for speed in [0.1, 0.04]:
    env=make_env(); obs,info=env.reset(seed=0)
    b,q,g=robot_state(obs); q0=q.copy()
    ys=[]
    for t in range(40):
        a=np.zeros(11,dtype=np.float32)
        a[1]=speed
        a[2]=np.clip(((np.pi-b[2]+np.pi)%(2*np.pi)-np.pi)/0.87,-0.1,0.1)
        dq=(q0-q+np.pi)%(2*np.pi)-np.pi; a[3:10]=np.clip(0.8*dq/0.249,-0.1,0.1)
        obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs); ys.append(round(b[2]-np.pi,3))
    print('basey speed',speed,'yawerr',ys[::5])
    env.close()
# test2: arm extends fast, base holds
for gain in [0.8, 0.25]:
    env=make_env(); obs,info=env.reset(seed=0)
    b,q,g=robot_state(obs); b0=b.copy(); b0[2]=np.pi
    qt=ik(np.array([0.62,0,0.16]), A.tool_R(0.0), q)
    ys=[];js=[]
    for t in range(80):
        a=make_action(b,q,b0,qt,0.0,gain=gain)
        obs,r,te,tr,i=env.step(a); b,q,g=robot_state(obs); ys.append(round(b[2]-np.pi,3)); js.append(round(float(np.max(np.abs((qt-q+np.pi)%(2*np.pi)-np.pi))),3))
    print('armgain',gain,'yawerr',ys[::10],'jerr',js[::10])
    env.close()
