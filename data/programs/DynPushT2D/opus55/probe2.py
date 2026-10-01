from env_client import make_env
import numpy as np
np.set_printoptions(precision=3, suppress=True)
env = make_env()
obs,_ = env.reset(seed=0)
# robot at 1.637,1.27; block at 1.05,3.44 theta 2.593
# move robot toward walls to find bounds: move left for 60 steps
for i in range(60):
    obs,r,te,tr,info=env.step(np.clip(np.array([-0.05,0.0]),-0.0499,0.0499))
print('left', obs[16:18])
for i in range(80):
    obs,r,te,tr,info=env.step(np.clip(np.array([0.0,-0.05]),-0.0499,0.0499))
print('down', obs[16:18])
for i in range(120):
    obs,r,te,tr,info=env.step(np.clip(np.array([0.05,0.0]),-0.0499,0.0499))
print('right', obs[16:18], obs[0:6])
# big action out of bounds
obs,r,te,tr,info=env.step(np.clip(np.array([0.2,0.2]),-0.0499,0.0499))
print('clip', obs[16:18], r, info)
