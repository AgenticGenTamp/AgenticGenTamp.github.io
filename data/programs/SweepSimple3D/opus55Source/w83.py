import sys,numpy as np, mujoco
from run_test import env
from approach import GeneratedApproach as G
seed=int(sys.argv[1]); n=int(sys.argv[2]); T=int(sys.argv[3])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=G(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(T): s,*_=env.step(ap.get_action(s))
oc=env._current_backend._object_centric_env; m=oc._robot_env.sim.model.mj_model; d=oc._robot_env.sim.data.mj_data
b,arm,g,cubes,_=ap._read(s)
print('wiper',ap.wiper.round(3)); print(ap.wiper_R.round(2)); print('cubes',np.round(cubes,3)); print('base',b.round(3))
for gi in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[gi])
    if 'wiper' in bn: print('geom',d.geom_xpos[gi].round(3), d.geom_xmat[gi].reshape(3,3).round(2).tolist())
