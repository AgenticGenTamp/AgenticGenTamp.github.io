import sys, numpy as np
from env_client import make_env
seeds=[int(s) for s in sys.argv[1].split(',')]
env=make_env(); os_=env.observation_space; T=os_.get_type
z=np.zeros(11,dtype=np.float32)
for sd in seeds:
    obs,info=env.reset(seed=sd)
    cs=[(round(float(obs.get(o,'x')),3),round(float(obs.get(o,'y')),3)) for o in obs.get_objects(T("mujoco_movable_object")) if o.name.startswith('cube')]
    obs,r,te,tr,inf=env.step(z)
    print("seed",sd,"r",repr(float(r)),"cubes",cs,flush=True)
env.close()
