import numpy as np, kin, ctrl
from env_client import make_env
qd=np.array([-0.01,1.77,3.16,-1.79,0.01,0.94,1.54])
env=make_env(); o,i=env.reset(seed=0,options={'object_count':1})
bias=np.zeros(7)
for t in range(400):
    q,b=ctrl.read_robot(o)
    err=qd-q
    if np.max(np.abs(err))<0.25: bias+=0.05*err
    bias=np.clip(bias,-0.1,0.1)
    a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(err/0.25+bias,-0.1,0.1)
    o,r,te,tr,inf=env.step(a)
    if t%50==0 or t==399: print(t, np.round(np.max(np.abs(err)),4), np.round(bias,3))
q,b=ctrl.read_robot(o); T=ctrl.ee_world(q,b)
print("ee rel base",np.round(T[:3,3]-np.array([b[0],b[1],0]),4),"want",np.round(ctrl.MOUNT+np.array([0.45,0,0]),3))
print("R",np.round(T[:3,:3],3))
env.close()
