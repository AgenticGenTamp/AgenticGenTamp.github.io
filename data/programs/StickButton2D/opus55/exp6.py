from env_client import make_env
import numpy as np, math
env=make_env()
obs,info=env.reset(seed=1)
r=obs.get_object_from_name('robot'); b=obs.get_object_from_name('button1')
def step(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=np.float32))
for i in range(40):
    th=obs.get(r,'theta'); d=np.clip(math.atan2(math.sin(-math.pi/2-th),math.cos(-math.pi/2-th)),-0.196,0.196)
    dx=np.clip(1.919-obs.get(r,'x'),-0.05,0.05); dy=np.clip(0.42-obs.get(r,'y'),-0.05,0.05)
    step([dx,dy,d,0.1,0])
print([round(obs.get(r,f),3) for f in ['x','y','theta','arm_joint']], obs.get(b,'color_g'))
for y in [0.40,0.38,0.37,0.36,0.35]:
    step([0,y-obs.get(r,'y'),0,0,0]); print(round(obs.get(r,'y'),3), obs.get(b,'color_g'))
