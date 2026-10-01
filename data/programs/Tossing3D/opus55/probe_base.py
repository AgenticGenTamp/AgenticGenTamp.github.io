from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=0)
sp=env.observation_space
r=obs.get_object_from_name('robot')
def st(o):
    r=o.get_object_from_name('robot')
    return np.array([o.get(r,'pos_base_x'),o.get(r,'pos_base_y'),o.get(r,'pos_base_rot')])
print('init',st(obs))
def step(a3,n=1):
    global obs
    rs=[]
    for _ in range(n):
        a=np.zeros(18,np.float32); a[:3]=a3
        obs,rw,te,tr,inf=env.step(a); rs.append(rw)
    return rs
# pulse test
s0=st(obs); step([0.1,0,0]); s1=st(obs); step([0,0,0],10); s2=st(obs)
print('pulse after1',s1-s0,'after10 zeros',s2-s0)
# steady speed
prev=st(obs)
for i in range(8):
    step([0.1,0,0]); s=st(obs); print('v',i,s-prev); prev=s
for i in range(5):
    step([0,0,0]); s=st(obs); print('settle',i,s-prev); prev=s
# rotate to pi/2
for i in range(30):
    s=st(obs); d=np.clip(np.pi/2-s[2],-0.1,0.1); step([0,0,d])
print('rot',st(obs))
s0=st(obs); step([0.1,0,0],5); print('x cmd at yaw pi/2 disp',st(obs)-s0)
s0=st(obs); step([0,0.1,0],5); print('y cmd at yaw pi/2 disp',st(obs)-s0)
env.close()
