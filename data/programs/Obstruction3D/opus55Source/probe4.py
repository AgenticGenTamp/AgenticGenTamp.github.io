import time, numpy as np
from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
t=time.time()
sim = ObjectCentricObstruction3DEnv(num_obstructions=3, allow_state_access=True)
print('construct', time.time()-t)
s,_ = sim.reset(seed=1)
t=time.time()
for i in range(20): sim.set_state(s)
print('set_state', (time.time()-t)/20)
t=time.time()
for i in range(50):
    a = np.zeros(11, dtype=np.float32); a[3]=0.1
    o = sim.step(a)
print('step', (time.time()-t)/50, type(o[0]), len(o))
print(sim.action_space)
import inspect
print(inspect.signature(sim.step))
