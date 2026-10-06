import sys, numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
pads=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'robot_'+g) for g in ['left_pad1','right_pad1']]
for t in range(int(sys.argv[3])):
    a=ap.get_action(s); nm=ap.queue[0].name
    s,r,term,tr,_=env.step(a)
    if nm in ('lower','close','lift'):
        b,arm,g,cubes,_=ap._read(s)
        sp=d.site_xpos[site]; c=min(cubes,key=lambda c: np.hypot(*(c[:2]-sp[:2])))
        print(t,nm,'site',sp.round(3),'pads',[d.geom_xpos[p].round(3).tolist() for p in pads],'cube',c.round(3), 'th', b[2].round(2))
    if term: break
