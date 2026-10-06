import sys, numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2]); T0,T1=int(sys.argv[3]),int(sys.argv[4])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
gn=lambda g: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ('body:'+str(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])))
for t in range(T1):
    a=ap.get_action(s); nm=ap.queue[0].name
    s,r,term,tr,_=env.step(a)
    if t>=T0:
        cs=set()
        for i in range(d.ncon):
            c=d.contact[i]; cs.add((gn(c.geom1),gn(c.geom2)))
        cs=[c for c in cs if 'robot' in c[0]+c[1] or 'pad' in c[0]+c[1] or 'cube' in c[0]+c[1]]
        print(t,nm,cs)
