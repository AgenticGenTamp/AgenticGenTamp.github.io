import sys, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
for g in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
    if 'wiper' in bn:
        print(bn, m.geom_pos[g], m.geom_quat[g], m.geom_size[g], m.geom_type[g])
        b=m.geom_bodyid[g]
print('body xpos', d.xpos[b], d.xquat[b], 'obs', obs[147:154])
