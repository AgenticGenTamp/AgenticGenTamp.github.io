import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env=make_env()
obs,_=env.reset(seed=0); p0=np.asarray(obs)[93:96].copy()
a=np.zeros(11,dtype=np.float32); a[1]=0.1
for t in range(10): obs,_,_,_,_=env.step(a)
p1=np.asarray(obs)[93:96].copy()
z=np.zeros(11,dtype=np.float32)
for t in range(20): obs,_,_,_,_=env.step(z)
p2=np.asarray(obs)[93:96].copy()
print("after10 push", p1-p0, "after 20 zero", p2-p0)
# rot settle
obs,_=env.reset(seed=0); q0=np.asarray(obs)[93:96].copy()
a=np.zeros(11,dtype=np.float32); a[2]=0.1
for t in range(10): obs,_,_,_,_=env.step(a)
q1=np.asarray(obs)[93:96].copy()
for t in range(20): obs,_,_,_,_=env.step(z)
q2=np.asarray(obs)[93:96].copy()
print("rot after10", q1-q0, "after zero", q2-q0)
# combined base+arm
obs,_=env.reset(seed=0); r0=np.asarray(obs)[[93,94,95,96]].copy()
a=np.zeros(11,dtype=np.float32); a[1]=0.1; a[3]=0.1
for t in range(10): obs,_,_,_,_=env.step(a)
print("combined delta base_y,j1:", np.asarray(obs)[[93,94,95,96]]-r0)
# out-of-range action clipping
obs,_=env.reset(seed=0); s0=np.asarray(obs)[[93,96]].copy()
a=np.zeros(11,dtype=np.float32); a[1]=0.5; a[3]=1.0
for t in range(10): obs,_,_,_,_=env.step(a)
print("oob action a1=0.5,a3=1.0 -> dy,dj1:", np.asarray(obs)[[94,96]]-np.array([0.0221,0.0]))
env.close()
