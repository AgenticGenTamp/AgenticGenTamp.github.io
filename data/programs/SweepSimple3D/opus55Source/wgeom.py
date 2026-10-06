import numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach as G
s,info=env.reset(seed=16,options={'object_count':1})
ap=G(env.action_space,env.observation_space,{}); ap.reset(s,info); ap._read(s)
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
print('wiper state',ap.wiper.round(3)); print(ap.wiper_R.round(2))
for g in range(m.ngeom):
    b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g])
    if 'wiper' in b:
        R=d.geom_xmat[g].reshape(3,3); e=np.abs(R)@m.geom_size[g] if m.geom_type[g]==6 else m.geom_size[g]
        print(m.geom_type[g], 'pos',d.geom_xpos[g].round(3),'size',m.geom_size[g].round(3),'aabb',(d.geom_xpos[g]-e).round(3),(d.geom_xpos[g]+e).round(3))
print('cube', ap._read(s)[3])
