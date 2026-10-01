import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=1)
def g(o,f): return round(float(obs.get(obs.get_object_from_name(o),f)),3)
for i in range(60):
    obs,r,term,trunc,info = env.step(np.array([0,0.049,0,0,0]))
    if i%5==0: print(i, g('robot','x'), g('robot','y'), r, term)
for i in range(30):
    obs,r,term,trunc,info = env.step(np.array([0,0,0.064,0,0]))
print('theta', g('robot','theta'))
for i in range(20):
    obs,r,term,trunc,info = env.step(np.array([0,0,0,0.099,0]))
    if i%3==0: print('arm', g('robot','arm_length'))
