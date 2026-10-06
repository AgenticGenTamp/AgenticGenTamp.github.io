import numpy as np
from common import make_env, robot
from approach import GeneratedApproach
env=make_env([1,2]); appr=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=0, options={'object_count':1}); appr.reset(s,info)
orig=appr._go
def go(bt,qt,g,max_n,**kw):
    i=0
    for a in orig(bt,qt,g,max_n,**kw):
        i+=1; yield a
    base,q=appr._robot(); print('go done',i,'max',max_n,'qerr',(qt-q).round(3),'berr',(bt-base).round(3))
appr._go=go
for i in range(120):
    s,*_=env.step(appr.get_action(s))
import mujoco
s,info=env.reset(seed=0, options={'object_count':1}); appr.reset(s,info)
for i in range(60): s,*_=env.step(appr.get_action(s))
sim=env._current_backend._object_centric_env._robot_env.sim; m=sim.model.mj_model; d=sim.data.mj_data
print('q', robot(s)[1].round(3))
for c in d.contact[:d.ncon]:
    b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]); b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2]); print('contact',b1,b2)
