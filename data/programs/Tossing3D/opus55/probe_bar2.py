from env_client import make_env
import numpy as np
env=make_env()
def st(o):
    r=o.get_object_from_name('robot')
    return np.array([o.get(r,'pos_base_x'),o.get(r,'pos_base_y'),o.get(r,'pos_base_rot'),o.get(r,'vel_base_x')])
obs,_=env.reset(seed=0)
for i in range(16):
    a=np.zeros(18,np.float32); a[0]=0.1
    obs,rw,te,tr,inf=env.step(a); print(i,st(obs).round(4))
# joint-limit vs collision: at yaw pi/2 earlier reached 1.018; try yaw pi/4
obs,_=env.reset(seed=0)
for i in range(20):
    s=st(obs); a=np.zeros(18,np.float32); a[2]=np.clip(np.pi/4-s[2],-.1,.1); obs,*_=env.step(a)
for i in range(20):
    a=np.zeros(18,np.float32); a[0]=0.1; obs,*_=env.step(a)
print('yaw pi/4 stop',st(obs).round(4))
# truncation
obs,_=env.reset(seed=0); n=0
while True:
    obs,rw,te,tr,inf=env.step(np.zeros(18,np.float32)); n+=1
    if te or tr: print('end at',n,te,tr,rw); break
    if n>2000: print('no end'); break
env.close()
