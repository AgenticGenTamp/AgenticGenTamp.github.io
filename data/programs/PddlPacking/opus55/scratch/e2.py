from env_client import make_env
import numpy as np
env=make_env(); T=env.observation_space.get_type
f=env.observation_space.type_features[T("robot")]
def rs(o):
    r=o.get_objects(T("robot"))[0]; return np.array([o.get(r,x) for x in f])
def step(d):
    a=np.zeros(11,np.float32); a[:len(d)]=d; o,*_=env.step(a); return o
# block on plate?
c=0
for s in range(301):
    o,_=env.reset(seed=s)
    for b in o.get_objects(T("block")):
        if abs(o.get(b,"pose_x"))<0.17 and abs(o.get(b,"pose_y"))<0.17: c+=1
print("blocks near plate",c)
# base limits: drive away from table
for d in [(-0.2,0,0),(0,0.2,0),(0,-0.2,0)]:
    o,_=env.reset(seed=0)
    for i in range(100): o=step(d)
    print(d, rs(o)[:3].round(4))
# rotate
o,_=env.reset(seed=0)
for i in range(20): o=step((0,0,0.2))
print("rot", rs(o)[:3].round(4))
o=step((0,0,-0.3)); print("clip rot", rs(o)[:3].round(4))
