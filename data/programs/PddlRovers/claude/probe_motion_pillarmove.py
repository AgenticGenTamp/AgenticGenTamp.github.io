from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def opos(obs,n):
    o=obs.get_object_from_name(n); return np.array([float(obs.get(o,'x')),float(obs.get(o,'y')),float(obs.get(o,'z'))])
names=['obstacle9','obstacle6','sample4','lander']
print("start", {n:np.round(opos(obs,n),4).tolist() for n in names})
px,py=0.654,-0.199
obs,ok=goto(env,obs,px+0.6,py,0)
print("after goto", np.round(opos(obs,'obstacle9'),4))
obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005)
print("blocked at relx=%.5f pillar now %s"%(p[0]-px, np.round(opos(obs,'obstacle9'),4)))
# push hard 20 times
for _ in range(20): obs,m,d=try_move(env,obs,-0.2,0,0,0)
print("after 20 pushes pos=%s pillar=%s"%(np.round(pose(obs,0),4), np.round(opos(obs,'obstacle9'),4)))
# exact pillar pos with more digits from raw state
st=None

env.close()
