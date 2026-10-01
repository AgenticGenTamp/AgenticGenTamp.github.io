from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
f=env.observation_space.type_features[T("robot")]
def rs(o):
    r=o.get_objects(T("robot"))[0]; return np.array([o.get(r,x) for x in f])
def step(d):
    a=np.zeros(11,np.float32); a[:len(d)]=d; o,*_=env.step(a); return o
print(env.action_space)
for i in range(10):
    d=np.zeros(10); d[i]=0.5 if i!=0 else -0.5
    o,_=env.reset(seed=0); s0=rs(o); o=step(d); print(i, (rs(o)-s0)[:10].round(4))
# x far: go to y=2, then x=+5
o,_=env.reset(seed=0)
for i in range(20): o=step((0,0.2,0))
for i in range(60): o=step((0.2,0,0))
print("far +x",rs(o)[:3])
