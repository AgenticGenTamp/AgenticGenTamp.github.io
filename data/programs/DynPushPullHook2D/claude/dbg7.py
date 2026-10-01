import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from ctl import rget, oget, act
env = make_env(); ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=3); ap.reset(obs,info)
for i in range(3000):
    a=ap.get_action(obs)
    if ap.phase=='grasp_close': break
    obs,r,term,tr,info = env.step(a)
print("hook vel", oget(obs,'hook','vx'), oget(obs,'hook','vy'), oget(obs,'hook','omega'))
for _ in range(40): obs,_,_,_,_=env.step(act())
print("after wait vel", oget(obs,'hook','vx'), oget(obs,'hook','vy'), oget(obs,'hook','omega'))
for k in range(12):
    obs,_,_,_,_=env.step(act(dg=-0.02))
    if oget(obs,'hook','held')>0.5: print("HELD at",rget(obs,'finger_gap')); break
else: print("still fail")
# now back off and re-approach along u fresh
hth=oget(obs,'hook','theta'); u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
print("hook",oget(obs,'hook','x'),oget(obs,'hook','y'),hth)
b=np.array([rget(obs,'x'),rget(obs,'y')])
fe=np.array([oget(obs,'hook','x'),oget(obs,'hook','y')])-1.4*u-0.0533*nv
print("long",(fe-b).dot(u),"lat",(fe-b).dot(nv),"arm",rget(obs,'arm_joint'))
env.close()
