import numpy as np
from env_client import make_env
def rob(obs):
    r=obs.get_object_from_name("robot"); f=obs.type_features[r.type]; d=obs.data[r]
    return {n:float(d[f.index(n)]) for n in f}
env=make_env(); obs,_=env.reset(seed=0)
# move joint4 toward 0 first, then push joint2 +
a=np.zeros(11,np.float32); a[6]=0.1
for _ in range(110): obs,*_=env.step(a)
print("joint4 now %.3f"%rob(obs)['pos_arm_joint4'])
a=np.zeros(11,np.float32); a[4]=0.1
prev=rob(obs)['pos_arm_joint2']
for i in range(200):
    obs,*_=env.step(a); q=rob(obs)['pos_arm_joint2']
print("joint2 + from alt config final=%.4f"%q)
env.close()
# gripper actuation: watch vel_gripper after cmd change
env=make_env(); obs,_=env.reset(seed=0)
a=np.zeros(11,np.float32); a[10]=1.0
vs=[]
for i in range(30):
    obs,*_=env.step(a); s=rob(obs); vs.append((round(s['pos_gripper'],3),round(s['vel_gripper'],4)))
print("close: (pos,vel) first10",vs[:10]); print("  steps10-30 vel",[v for _,v in vs[10:30:2]])
env.close()
