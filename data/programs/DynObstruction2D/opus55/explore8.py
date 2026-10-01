from env_client import make_env
import numpy as np, sys
from util import *
env = make_env()
seed=int(sys.argv[1]); X=float(sys.argv[2]); G=float(sys.argv[3])
obs, info = env.reset(seed=seed)
obs,_ = goto(env, obs, [1.0, 1.4, None, 0.24, G])
obs,_ = goto(env, obs, [None, None, -np.pi/2, None, None])
obs,_ = goto(env, obs, [X, None, None, 0.48, None])
print(objs(obs)['target_block'])
for i in range(16):
    obs,*_ = env.step(np.array([0,-0.03,0,0,0]))
    print('down',rob(obs).round(3), objs(obs)['target_block'])
for i in range(10):
    obs,*_ = env.step(np.array([0.03,-0.01,0,0,0]))
    print('drag',rob(obs).round(3), objs(obs)['target_block'])
env.close()
