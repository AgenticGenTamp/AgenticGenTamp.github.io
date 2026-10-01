from env_client import make_env
import numpy as np
from util import *
env = make_env()
obs, info = env.reset(seed=0)
obs,_ = goto(env, obs, [None, 1.3, None, 0.24, 0.12])
obs,_ = goto(env, obs, [None, None, -np.pi/2, None, None])
obs,_ = goto(env, obs, [1.526-0.214-0.2, None, None, None, None])
obs,_ = goto(env, obs, [None, None, None, 0.48, None])
obs,_ = goto(env, obs, [None, 0.83, None, None, None])
print('lowered', rob(obs).round(3), objs(obs)['target_block'])
for sp in [0.01, 0.03, 0.049]:
  for i in range(8):
    obs,*_ = env.step(np.array([sp,0,0,0,0]))
    B=obs.get_object_from_name('target_block')
    print('push', rob(obs).round(3), objs(obs)['target_block'], 'gapL', round(obs.get(B,'x')-0.214-rob(obs)[0],3))
  for i in range(4):
    obs,*_ = env.step(np.array([0,0,0,0,0]))
    print('wait', rob(obs).round(3), objs(obs)['target_block'])
env.close()
