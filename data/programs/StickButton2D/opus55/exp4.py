from env_client import make_env
import numpy as np, math
env=make_env()
obs,info=env.reset(seed=357)
r=obs.get_object_from_name('robot')
def rs(): return [round(float(obs.get(r,f)),4) for f in ['x','y','theta','arm_joint']]
def step(a):
    global obs
    obs,*_=env.step(np.array(a,dtype=np.float32))
for i in range(40):
    th=obs.get(r,'theta'); d=np.clip(math.atan2(math.sin(1.28-th),math.cos(1.28-th)),-0.196,0.196)
    step([0,0,d,0,0])
for i in range(60):
    step([min(0.05,3.38-obs.get(r,'x')),min(0.05,max(-0.05,0.3-obs.get(r,'y'))),0,0,0])
print(rs())
step([0,0,0,0.1,0]); print('ext',rs())
for i in range(40):
    y=obs.get(r,'y'); step([0,0.005,0,0,0])
    if abs(obs.get(r,'y')-y)<1e-6: print('blocked up at',rs()); break
for i in range(10):
    x=obs.get(r,'x'); step([0.005,0,0,0,0])
    if abs(obs.get(r,'x')-x)<1e-6: print('blocked right at',rs()); break
