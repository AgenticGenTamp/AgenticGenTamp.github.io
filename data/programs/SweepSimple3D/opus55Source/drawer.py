import sys, numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2]); T=int(sys.argv[3])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
js=[j for j in range(m.njnt) if 'drawer' in str(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j))]
print([(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j), m.jnt_type[j]) for j in js])
last=None
for t in range(T):
    a=ap.get_action(s); nm=ap.queue[0].name
    s,*_=env.step(a)
    v=tuple(round(float(d.qpos[m.jnt_qposadr[j]]),3) for j in js)
    if v!=last: print(t,nm,v, ap._read(s)[0].round(3)); last=v
