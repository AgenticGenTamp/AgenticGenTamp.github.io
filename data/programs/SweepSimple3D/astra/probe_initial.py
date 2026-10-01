from env_client import make_env
import numpy as np
E=make_env()
for seed in range(3):
 s,i=E.reset(seed=seed)
 print('SEED',seed,'INFO',i, 'MAX',E.max_steps,flush=True)
 for typ in E.observation_space.types:
  if typ.name not in ("mujoco_tidybot_robot", "mujoco_fixture"): continue
  objs=s.get_objects(typ)
  if objs:
   print('TYPE',typ,flush=True)
   for o in objs:
    print(o, {f:s.get(o,f) for f in E.observation_space.type_features[typ]},flush=True)
 a=np.zeros(11)
 s,r,t,tr,i=E.step(a)
 print('ZERO STEP',r,t,tr,i,flush=True)
E.close()
