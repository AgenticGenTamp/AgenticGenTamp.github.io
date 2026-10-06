import sys, numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2]); T=int(sys.argv[3])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
ap._read(s); w0=ap.wiper.copy()
for t in range(T):
    a=ap.get_action(s); nm=ap.queue[0].name
    s,r,term,*_=env.step(a); ap._read(s)
    if np.hypot(*(ap.wiper[:2]-w0[:2]))>0.005:
        cons=set()
        for k in range(d.ncon):
            cc=d.contact[k]; b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[cc.geom1]); b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[cc.geom2])
            if ('wiper' in b1+b2 or 'cube_4' in b1+b2) and 'world' not in b1+b2 : cons.add(b1+'/'+b2)
        print(t,nm,'wiper',w0[:2].round(3),'->',ap.wiper[:2].round(3),'base',ap._read(s)[0].round(2),sorted(cons)[:4])
    w0=ap.wiper.copy()
    if term: break
