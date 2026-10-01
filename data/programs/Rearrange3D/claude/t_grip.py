import numpy as np, time
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
obs,_=env.reset(seed=0); print("init grip", np.asarray(obs)[103])
a=np.zeros(11,dtype=np.float32); a[10]=1.0
g=[]
for t in range(30):
    obs,_,_,_,_=env.step(a); g.append(round(float(np.asarray(obs)[103]),4))
print("cmd 1.0:", g)
a[10]=0.0
g=[]
for t in range(30):
    obs,_,_,_,_=env.step(a); g.append(round(float(np.asarray(obs)[103]),4))
print("cmd 0.0:", g)
a[10]=0.5
g=[]
for t in range(20):
    obs,_,_,_,_=env.step(a); g.append(round(float(np.asarray(obs)[103]),4))
print("cmd 0.5:", g)
# timing
obs,_=env.reset(seed=0)
t0=time.time()
for t in range(100):
    env.step(np.zeros(11,dtype=np.float32))
print("100 zero steps sec:", round(time.time()-t0,3))
t0=time.time()
rng=np.random.default_rng(0)
for t in range(100):
    env.step(rng.uniform(-0.1,0.1,11).astype(np.float32))
print("100 rand steps sec:", round(time.time()-t0,3))
env.close()
