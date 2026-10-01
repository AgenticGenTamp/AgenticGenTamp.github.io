from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r = obs.get_objects(env.observation_space.get_type('crv_robot'))[0]
def g(): return [round(float(obs.get(r,f)),4) for f in ['x','y','arm_joint']]
for d,a in [('up',[0,.05,0,0,0]),('left',[-.05,0,0,0,0]),('right',[.05,0,0,0,0])]:
    for i in range(40): obs,*_ = env.step(np.array(a,dtype=np.float32))
    print(d, g())
# retract arm then go down
for i in range(3): obs,*_ = env.step(np.array([0,0,0,-.1,0],dtype=np.float32))
for i in range(40): obs,*_ = env.step(np.array([0,-.05,0,0,0],dtype=np.float32))
print('down retract', g())
print(env.max_steps)
