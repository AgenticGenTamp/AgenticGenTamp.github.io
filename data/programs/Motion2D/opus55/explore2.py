from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r = obs.get_object_from_name('robot')
def p(o): return tuple(round(float(o.get(r,f)),4) for f in ['x','y','theta','arm_joint'])
print(p(obs))
# rotate toward 0
for i in range(20):
    obs,rew,term,trunc,_ = env.step(np.array([0,0,0.196,0,0],dtype=np.float32))
print('rot', p(obs))
for i in range(5):
    obs,rew,term,trunc,_ = env.step(np.array([0,0,0,-0.1,0],dtype=np.float32))
print('retract', p(obs))
for i in range(5):
    obs,rew,term,trunc,_ = env.step(np.array([0,0,0,0.1,0],dtype=np.float32))
print('extend', p(obs))
for i in range(10):
    obs,rew,term,trunc,_ = env.step(np.array([0,0.05,0,0,0],dtype=np.float32))
    print('up', p(obs), rew, term)
for i in range(10):
    obs,rew,term,trunc,_ = env.step(np.array([0.05,0.0,0,0,0],dtype=np.float32))
    print('right', p(obs), rew, term)
for i in range(10):
    obs,rew,term,trunc,_ = env.step(np.array([-0.05,0.0,0,0,0],dtype=np.float32))
    print('left', p(obs), rew, term)
env.close()
