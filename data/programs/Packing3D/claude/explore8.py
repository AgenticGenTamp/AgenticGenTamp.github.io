import numpy as np
from env_client import make_env
from lib_probe import *
# Test: what does finger_state do? try close then move, watch collisions. Also check base motion limits.
env=make_env(); obs,info=env.reset(seed=0)
r=obs.get_object_from_name('robot')
def rv(o,f): return round(float(o.get(o.get_object_from_name('robot'),f)),4)
# base motion sweep in x toward table
for i in range(12):
    a=np.zeros(11); a[0]=0.05
    prev=(rv(obs,'pos_base_x'),)
    obs,*_=env.step(a)
    print("bx",rv(obs,'pos_base_x'))
env.close()
