from env_client import make_env
import numpy as np
from util import *
env = make_env()
obs, info = env.reset(seed=11)  # no obstructions, block at 2.451 w .309
print(objs(obs))
obs,_ = goto(env, obs, [None, 1.3, None, 0.24, 0.32])
obs,_ = goto(env, obs, [5, None, None, None, None])
print('xmax', rob(obs).round(3))
obs,_ = goto(env, obs, [None, None, -np.pi/2, None, None])
obs,_ = goto(env, obs, [5, None, None, None, None])
print('xmax', rob(obs).round(3))
obs,_ = goto(env, obs, [None, 5, None, None, None])
print('ymax', rob(obs).round(3))
obs,_ = goto(env, obs, [2.451+0.155+0.25, 1.3, None, None, None])
obs,_ = goto(env, obs, [None, None, None, 0.48, None])
obs,_ = goto(env, obs, [None, 0.79, None, None, None])
print('lowered', rob(obs).round(3), objs(obs)['target_block'])
for i in range(80):
    obs,r,term,trunc,info = env.step(np.array([-0.02,0,0,0,0]))
    if i%5==0 or term: print(rob(obs).round(3), objs(obs)['target_block'], term)
    if term: break
env.close()
