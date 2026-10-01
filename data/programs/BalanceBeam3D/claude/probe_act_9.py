import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32); a[3]=0.1
for i in range(10): obs,r,te,tu,_=env.step(a)
q=obs[19]; b=obs[16:19].copy()
z=np.zeros(11,dtype=np.float32); tail=[]
for i in range(6): obs,r,te,tu,_=env.step(z); tail.append(float(obs[19]))
print("after stop, joint1 residual motion:", np.round(np.array(tail)-q,5), "base residual", (obs[16:19]-b).round(5))
# base stop residual
obs,_=env.reset(seed=0); a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(10): obs,r,te,tu,_=env.step(a)
b=obs[16:19].copy()
t=[]
for i in range(4): obs,r,te,tu,_=env.step(z); t.append(obs[16].copy())
print("base residual after stop:", np.round(np.array(t)-b[0],5))
env.close()
