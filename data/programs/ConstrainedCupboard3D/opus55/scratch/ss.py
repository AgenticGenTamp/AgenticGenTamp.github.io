import sys; sys.path.insert(0,'.')
import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=11, options={'object_count':1})
try:
    s=env.get_state(); print('get_state ok', type(s))
except Exception as e: print('get_state fail', repr(e)[:200])
M=env.observation_space.get_type('mujoco_movable_object')
o=obs.get_objects(M)[0]
s=obs.copy(); s.set(o,'x',2.0); s.set(o,'y',0.0); s.set(o,'z',0.3)
try:
    env.set_state(s); print('set_state ok')
    ob,r,te,tr,inf=env.step(np.zeros(11)); print('after', [round(ob.get(o,k),3) for k in 'xyz'], r, te)
except Exception as e: print('set_state fail', repr(e)[:300])
