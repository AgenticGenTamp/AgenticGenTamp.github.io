import sys, numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2]); T=int(sys.argv[3])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
prev=None
for t in range(T):
    a=ap.get_action(s); q=ap.queue[0]; nm=q.name
    s,r,term,tr,_=env.step(a)
    if nm=='close' and prev!='close':
        b,arm,g,cubes,_=ap._read(s); sp=d.site_xpos[site].copy()
        i=int(np.argmin([np.hypot(*(c[:2]-sp[:2])) for c in cubes])); c=cubes[i]
        print(t,'close site',sp.round(3),'cube',i,c.round(3),'base',b.round(3),'wiper',ap.wiper.round(3))
        cons=set()
        for k in range(d.ncon):
            cc=d.contact[k]; n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,cc.geom1); n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,cc.geom2)
            if 'robot' in str(n1)+str(n2): cons.add((n1,n2))
        print('   cons',list(cons)[:6])
    if nm=='lift' and prev=='close':
        pass
    prev=nm
    if term: break
