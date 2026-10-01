from env_client import make_env
import numpy as np, math
env=make_env()
obs,info=env.reset(seed=1)
r=obs.get_object_from_name('robot')
def step(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=np.float32))
for i in range(40):
    th=obs.get(r,'theta'); d=np.clip(math.atan2(math.sin(math.pi/2-th),math.cos(math.pi/2-th)),-0.196,0.196)
    dx=np.clip(1.0-obs.get(r,'x'),-0.05,0.05); dy=np.clip(1.14-obs.get(r,'y'),-0.05,0.05)
    step([dx,dy,d,0,0])
print([round(obs.get(r,f),3) for f in ['x','y','theta','arm_joint']])
step([0,0,0,0.1,0]); print([round(obs.get(r,f),3) for f in ['x','y','theta','arm_joint']])
# walls: go to left wall with arm pointing left
for i in range(40):
    th=obs.get(r,'theta'); d=np.clip(math.atan2(math.sin(math.pi-th),math.cos(math.pi-th)),-0.196,0.196)
    dx=np.clip(0.5-obs.get(r,'x'),-0.05,0.05); dy=np.clip(0.6-obs.get(r,'y'),-0.05,0.05)
    step([dx,dy,d,-0.1,0])
step([0,0,0,0.1,0]); print([round(obs.get(r,f),3) for f in ['x','y','theta','arm_joint']])
for i in range(60):
    x=obs.get(r,'x'); step([-0.005,0,0,0,0])
    if abs(obs.get(r,'x')-x)<1e-6: break
print('blocked',[round(obs.get(r,f),4) for f in ['x','y','theta','arm_joint']])
