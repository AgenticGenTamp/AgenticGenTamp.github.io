import sys
from env_client import make_env
import numpy as np
seed=int(sys.argv[1]); xoff=float(sys.argv[2]); ny=int(sys.argv[3])
env=make_env(); os_=env.observation_space; T=os_.get_type
obs,info=env.reset(seed=seed)
RF=os_.type_features[T("mujoco_tidybot_robot")]
def snap(obs):
    d={}
    for o in obs.get_objects(T("mujoco_movable_object")):
        d[o.name]=(round(float(obs.get(o,'x')),3),round(float(obs.get(o,'y')),3))
    r=obs.get_objects(T("mujoco_tidybot_robot"))[0]
    d['B']=(round(float(obs.get(r,'pos_base_x')),3),round(float(obs.get(r,'pos_base_y')),3))
    return d
s0=snap(obs); print("t0",s0,flush=True)
# align base x with wiper x + xoff
tgt=s0['wiper_0'][0]+xoff
for i in range(12):
    dx=np.clip(tgt-snap(obs)['B'][0],-0.1,0.1)
    if abs(dx)<0.002: break
    a=np.zeros(11,dtype=np.float32); a[0]=dx
    obs,r,te,tr,inf=env.step(a)
print("aligned",snap(obs),flush=True)
a=np.zeros(11,dtype=np.float32); a[1]=-0.1
prev=-1.0
for i in range(ny):
    obs,r,te,tr,inf=env.step(a); r=float(r)
    print(i,repr(r),snap(obs),te,tr,flush=True)
    if te or tr: break
env.close()
