import numpy as np
from env_client import make_env
J=["joint_%d"%i for i in range(1,8)]
def basev(o):
    r=o.get_object_from_name("robot"); return np.array([float(o.get(r,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]])
for idx,step in [(0,0.05),(0,-0.05),(1,0.05),(1,-0.05)]:
    env=make_env(); obs,_=env.reset(seed=79)
    prev=basev(obs)
    for _ in range(40):
        a=np.zeros(11); a[idx]=step
        o2,*_=env.step(a); v=basev(o2)
        if np.allclose(v,prev,atol=1e-9): break
        prev=v
    print("idx",idx,"step",step,"limit",np.round(prev,3))
    env.close()
# combined: move y then x
env=make_env(); obs,_=env.reset(seed=79)
prev=basev(obs)
for _ in range(20):
    a=np.zeros(11); a[1]=-0.05
    o2,*_=env.step(a); prev=basev(o2)
print("after y move",np.round(prev,3))
for _ in range(40):
    a=np.zeros(11); a[0]=0.05
    o2,*_=env.step(a); v=basev(o2)
    if np.allclose(v,prev,atol=1e-9): break
    prev=v
print("then x limit",np.round(prev,3))
env.close()
