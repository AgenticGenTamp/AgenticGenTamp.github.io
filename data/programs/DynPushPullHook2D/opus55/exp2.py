import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=1)
R=obs.get_object_from_name('robot')
def g(o,f): return round(float(obs.get(obs.get_object_from_name(o),f)),3)
def step(a):
    global obs
    obs,r,t,tr,info=env.step(np.clip(np.array(a,float),env.action_space.low*0.99,env.action_space.high*0.99)); return t
for i in range(20): step([0.05,0.05,0,0,0])
print('pos',g('robot','x'),g('robot','y'),g('robot','theta'))
for i in range(10): step([0,0,-0.065,0,0])
print('theta',g('robot','theta'))
for i in range(20):
    step([0,0,0,0.1,0])
print('arm', g('robot','arm_length'))
for i in range(30):
    step([0,0,0,0,0.02])
print('gap', g('robot','finger_gap'))
for i in range(30):
    step([0,0,0,0,-0.02])
print('gap', g('robot','finger_gap'))
for i in range(30):
    step([0,0,0,-0.1,0])
print('arm', g('robot','arm_length'))
