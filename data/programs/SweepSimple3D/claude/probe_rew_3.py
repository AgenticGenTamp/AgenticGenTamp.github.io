import sys,time
from env_client import make_env
import numpy as np
dirx,diry=float(sys.argv[1]),float(sys.argv[2]); nst=int(sys.argv[3]); seed=int(sys.argv[4])
env=make_env(); os_=env.observation_space; T=os_.get_type
obs,info=env.reset(seed=seed)
def snap(obs):
    d={}
    for o in obs.get_objects(T("mujoco_movable_object")):
        d[o.name]=(round(float(obs.get(o,'x')),3),round(float(obs.get(o,'y')),3),round(float(obs.get(o,'z')),3))
    r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
    d['BASE']=(round(float(obs.get(r,'pos_base_x')),3),round(float(obs.get(r,'pos_base_y')),3))
    return d
print("t0",snap(obs),flush=True)
a=np.zeros(11,dtype=np.float32); a[0]=dirx*0.1; a[1]=diry*0.1
t=time.time()
for i in range(nst):
    obs,r,te,tr,inf=env.step(a)
    print(i,round(float(r),4),snap(obs),te,tr,round(time.time()-t,1),flush=True)
    if te or tr: break
env.close()
