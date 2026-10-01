import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from ctl import rget, oget, act
env = make_env(); ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=3); ap.reset(obs,info)
for i in range(3000):
    a=ap.get_action(obs)
    if ap.phase=='grasp_close': break
    obs,r,term,tr,info = env.step(a)
print("phase",ap.phase,"d_grasp",ap.d_grasp,"i",i)
print("robot",[round(rget(obs,f),4) for f in ('x','y','theta','arm_joint','finger_gap')])
hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
print("hook",round(hx,4),round(hy,4),round(hth,4))
u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
fe=np.array([hx,hy])-1.4*u-0.0533*nv
b=np.array([rget(obs,'x'),rget(obs,'y')])
rel=fe-b
print("fe",np.round(fe,4),"long",round(rel.dot(u),4),"lat",round(rel.dot(nv),4))
# close step by step
for k in range(12):
    obs,r,term,tr,info=env.step(act(dg=-0.02))
    print(k, round(rget(obs,'finger_gap'),3), "held",oget(obs,'hook','held'), "hook",round(oget(obs,'hook','x'),4),round(oget(obs,'hook','y'),4),round(oget(obs,'hook','theta'),4))
    if oget(obs,'hook','held')>0.5: break
env.close()
