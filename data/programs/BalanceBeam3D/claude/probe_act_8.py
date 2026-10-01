import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
# rotate base ~90deg then move +x: world or body frame?
env=make_env(); obs,_=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32); a[2]=0.1
for i in range(16): obs,r,te,tu,_=env.step(a)
print("after yaw:", obs[16:19])
b0=obs[16:19].copy()
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(5): obs,r,te,tu,_=env.step(a)
print("dx,dy after +x cmd:", (obs[16:19]-b0).round(4), "-> body-frame" if abs(obs[17]-b0[1])>0.1 else "-> world-frame")
env.close()
# simultaneous base+arm+gripper
env=make_env(); obs,_=env.reset(seed=0)
q0=obs[19:26].copy(); b0=obs[16:19].copy()
a=np.array([0.1,0.1,0.05,0.1,0,0,0.1,0,0,0,1.0],dtype=np.float32)
for i in range(10): obs,r,te,tu,_=env.step(a)
print("combined: dbase",(obs[16:19]-b0).round(4)/10,"dq",((obs[19:26]-q0)/10).round(4),"grip",obs[26])
env.close()
# reward structure check: does reward ever differ from -1?
env=make_env(); obs,_=env.reset(seed=0)
rs=[]
rng=np.random.default_rng(0)
for i in range(200):
    a=np.concatenate([rng.uniform(-0.1,0.1,10),[rng.uniform(0,1)]]).astype(np.float32)
    obs,r,te,tu,_=env.step(a); rs.append(r)
    if te or tu: print("ended",i,te,tu); break
print("random rollout rewards: min",min(rs),"max",max(rs),"uniq",np.unique(np.round(rs,3))[:6])
env.close()
