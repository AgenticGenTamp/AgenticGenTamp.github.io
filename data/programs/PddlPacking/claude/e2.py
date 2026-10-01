from env_client import make_env
import numpy as np
F = ["base_x","base_y","base_rot"]
def rob(s):
    r = s.get_object_from_name("robot")
    return np.array([s.get(r,f) for f in F])
env = make_env()
obs,info = env.reset(seed=0)
# drive base +x until rejected
prev=rob(obs)
for k in range(20):
    a=np.zeros(11); a[0]=0.05
    obs,_,_,_,_=env.step(a)
    c=rob(obs)
    if np.allclose(c,prev): print("blocked at",np.round(c,4)); break
    prev=c
print("after x push", np.round(rob(obs),4))
# rotate base 90deg then push +x
for k in range(10):
    obs,_,_,_,_=env.step(np.array([0,0,0.15707963,0,0,0,0,0,0,0,0]))
print("after rot", np.round(rob(obs),4))
before=rob(obs)
obs,_,_,_,_=env.step(np.array([0.1,0,0,0,0,0,0,0,0,0,0]))
print("dx0.1 with rot:", np.round(rob(obs)-before,4))
env.close()
