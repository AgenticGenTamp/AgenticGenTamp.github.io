from env_client import make_env
import numpy as np
env = make_env()
T=env.observation_space.get_type
def rs(o):
    r=o.get_objects(T("robot"))[0]; f=env.observation_space.type_features[T("robot")]
    return np.array([o.get(r,x) for x in f])
for j in range(3,10):
  for sgn in [1,-1]:
    obs,_=env.reset(seed=0)
    # first lift arm: move base back for no collisions
    prev=None
    for k in range(40):
        a=np.zeros(11,dtype=np.float32); a[j]=0.2*sgn
        obs,*_=env.step(a); s=rs(obs)
        if prev is not None and abs(s[j]-prev)<1e-6: break
        prev=s[j]
    print("joint",j-2,"sgn",sgn,"final",round(s[j],4),"steps",k)
env.close()
