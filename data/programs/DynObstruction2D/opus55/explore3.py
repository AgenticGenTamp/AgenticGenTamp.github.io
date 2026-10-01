from env_client import make_env
import numpy as np
from util import *
env = make_env()
obs, info = env.reset(seed=0)
obs,_ = goto(env, obs, [None, 1.3, None, 0.24, 0.32])
obs,_ = goto(env, obs, [None, None, -np.pi/2, None, None])
obs,_ = goto(env, obs, [2.553, None, None, None, None])
print(rob(obs).round(3), objs(obs))
obs,_ = goto(env, obs, [None, None, None, 0.48, None])
obs,_ = goto(env, obs, [None, 0.3, None, None, None])
print('lowered', rob(obs).round(3), objs(obs))
for i in range(12):
    obs,*_ = env.step(np.array([0,0,0,0,-0.0199]))
    print('close', rob(obs).round(3), objs(obs)['obstruction1'])
for i in range(10):
    obs,*_ = env.step(np.array([0,0.049,0,0,0]))
    print('lift', rob(obs).round(3), objs(obs)['obstruction1'])
for i in range(10):
    obs,*_ = env.step(np.array([-0.049,0.0,0,0,0]))
    print('move', rob(obs).round(3), objs(obs)['obstruction1'])
env.close()
