import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
for g in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
    gn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
    if 'island' in bn and 's1c1' in bn+gn and m.geom_contype[g]+m.geom_conaffinity[g]>0:
        c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
        R=d.geom_xmat[g].reshape(3,3); p=d.geom_xpos[g]+R@c; e=np.abs(R)@h
        print(bn,gn,m.geom_type[g], (p-e).round(3),(p+e).round(3), m.geom_friction[g])
