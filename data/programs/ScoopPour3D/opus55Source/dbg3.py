import sys, numpy as np
from simutil import make_env, obj, robot_q
from approach import GeneratedApproach
N=10; env=make_env(N); ap=GeneratedApproach(env.action_space,None,{})
s,info=env.reset(seed=1,options={'object_count':N}); ap.reset(s,info)
for k in range(100):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
b,q=robot_q(s); bt,pos,R,g,_,_,_=ap.wps[ap.wi]
qi,err=ap.kin.ik(bt,q,pos,R); print('ikerr',err,'dq',np.round(qi-q,3),'I',np.round(ap.I,3),'a',np.round(a,3))
print('base',b,'bt',bt)
oc=env._current_backend._object_centric_env; d=oc._robot_env.sim.data.mj_data
import mujoco
m=oc._robot_env.sim.model.mj_model; sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
print('true site',d.site_xpos[sid],'fk',ap.kin.fk(b,q)[0])
print('true qpos base/arm',d.qpos[101:111])
