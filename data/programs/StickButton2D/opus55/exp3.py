from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=1)
r=obs.get_object_from_name('robot'); s=obs.get_object_from_name('stick')
def rs(obs): return [round(float(obs.get(r,f)),4) for f in ['x','y','theta','arm_joint','vacuum']]+[round(float(obs.get(s,f)),4) for f in ['x','y','theta']]
def step(a):
    global obs
    obs,rew,term,trunc,_=env.step(np.array(a,dtype=np.float32))
for i in range(30):
    th=obs.get(r,'theta'); d=np.clip(-np.arctan2(np.sin(th),np.cos(th)),-0.196,0.196)
    step([0,0,d,0.1,0])
for i in range(80):
    step([0.05,0,0,0,0])
print('right',rs(obs))
for i in range(10):
    step([0.005,0,0,0,0])
print('right',rs(obs))
step([0,0,0,0,1]); print('vac',rs(obs))
for i in range(3):
    step([-0.05,0,0,0,1]); print('left',rs(obs))
for i in range(3):
    step([0,0.05,0,0,1]); print('up',rs(obs))
for i in range(3):
    step([0,0,0.1,0,1]); print('rot',rs(obs))
for i in range(2):
    step([0,0,0,-0.05,1]); print('armin',rs(obs))
step([0,0,0,0,0]); print('vacoff',rs(obs))
step([-0.05,0,0,0,0]); print('left',rs(obs))
