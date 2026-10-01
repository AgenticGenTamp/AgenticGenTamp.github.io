from env_client import make_env
import numpy as np

env=make_env()
s,info=env.reset(seed=0)
print('INFO',info, 'LIMIT',env.max_steps, flush=True)
for t in env.observation_space.types:
    objs=s.get_objects(t)
    for o in objs:
        print(str(t), str(o), {f:s.get(o,f) for f in env.observation_space.type_features[t]},flush=True)
for i in range(3):
    s,r,te,tr,info=env.step(np.zeros(11))
    print('STEP',i,r,te,tr,info,flush=True)
env.close()
