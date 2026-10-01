import time
from env_client import make_env
import numpy as np
t0=time.time()
env=make_env(); obs,info=env.reset(seed=1)
print("setup %.1fs"%(time.time()-t0))
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32)); return rew
def P(tag): print(tag, round(R("pos_base_x"),4), round(R("pos_base_y"),4), round(R("pos_base_rot"),4))
P("init")
t0=time.time()
a=np.zeros(11); a[1]=-0.05
for i in range(10):
    step(a); P("y-%d"%i)
print("10 steps %.2fs"%(time.time()-t0))
a=np.zeros(11); a[0]=-0.05
for i in range(6): step(a); P("x-%d"%i)
a=np.zeros(11); a[2]=0.05
for i in range(6): step(a); P("yaw+%d"%i)
env.close()
