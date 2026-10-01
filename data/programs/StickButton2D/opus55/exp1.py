from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=0)
r=obs.get_object_from_name('robot')
def rs(obs): return [round(float(obs.get(r,f)),3) for f in ['x','y','theta','arm_joint']]
# move up
for i in range(30):
    obs,rew,term,trunc,_=env.step(np.array([0,0.05,0,0,0],dtype=np.float32))
print('up',rs(obs))
for i in range(80):
    obs,rew,term,trunc,_=env.step(np.array([0.05,0,0,0,0],dtype=np.float32))
print('right',rs(obs))
for i in range(80):
    obs,rew,term,trunc,_=env.step(np.array([-0.05,0,0,0,0],dtype=np.float32))
print('left',rs(obs))
for i in range(30):
    obs,rew,term,trunc,_=env.step(np.array([0,-0.05,0,0,0],dtype=np.float32))
print('down',rs(obs))
for i in range(5):
    obs,rew,term,trunc,_=env.step(np.array([0,0,0.1,0.1,0],dtype=np.float32))
    print('arm',rs(obs))
