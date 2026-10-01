import sys
from env_client import make_env
import numpy as np
idx=int(sys.argv[1]); sgn=float(sys.argv[2]); n=int(sys.argv[3])
env=make_env(); os_=env.observation_space; T=os_.get_type
obs,info=env.reset(seed=0)
RF=os_.type_features[T("mujoco_tidybot_robot")]
def snap(obs):
    d={}
    for o in obs.get_objects(T("mujoco_movable_object")):
        d[o.name]=(round(float(obs.get(o,'x')),3),round(float(obs.get(o,'y')),3),round(float(obs.get(o,'z')),3))
    r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
    d['ROB']=[round(float(obs.get(r,f)),2) for f in RF[:11]]
    return d
print("t0",snap(obs),flush=True)
a=np.zeros(11,dtype=np.float32); a[idx]=sgn*0.1
for i in range(n):
    obs,r,te,tr,inf=env.step(a)
    print(i,repr(float(r)),snap(obs),flush=True)
env.close()
