from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
env=make_env()
obs,_=env.reset(seed=42)
def step(a,n=1):
    global obs
    for _ in range(n):
        obs,r,t,tr,info=env.step(np.array(a,dtype=np.float32))
    return obs
print("init",obs[:9])
# try moving up a lot
for i in range(40):
    o=step([0,0.05,0,0,0])
print("after up40", o[:9])
for i in range(40):
    o=step([0.05,0,0,0,0])
print("after right40", o[:9])
for i in range(60):
    o=step([-0.05,-0.05,0,0,0])
print("after dl60", o[:9])
# arm extension
for i in range(20):
    o=step([0,0,0,0.1,0])
print("after arm out", o[:9])
for i in range(20):
    o=step([0,0,0,-0.1,0])
print("after arm in", o[:9])
for i in range(20):
    o=step([0,0,0.196,0,0])
print("after rot", o[:9])
env.close()
