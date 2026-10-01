import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=1)
def g(o,f): return round(float(obs.get(obs.get_object_from_name(o),f)),3)
def step(a):
    global obs
    obs,r,t,tr,info=env.step(np.clip(np.array(a,float),env.action_space.low*0.99,env.action_space.high*0.99)); return t
for i in range(10): step([0.03,0.03,0,0,0])
for d in [-0.1,-0.05,-0.02,0.02,0.05,0.1]:
    for i in range(3): step([0,0,0,d,0])
    print(d,'arm', g('robot','arm_length'), g('robot','arm_joint'))
for i in range(10): step([0,0,0,-0.1,0])
print('arm', g('robot','arm_length'), g('robot','arm_joint'))
