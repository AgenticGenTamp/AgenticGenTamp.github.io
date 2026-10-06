import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
print('wiper obs',obs[147:163].round(3))
for g in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
    if 'wiper' in bn or 'cube_0' in bn:
        c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
        R=d.geom_xmat[g].reshape(3,3); p=d.geom_xpos[g]+R@c; e=np.abs(R)@h
        print(bn,m.geom_type[g],(p-e).round(4),(p+e).round(4),m.geom_friction[g],m.geom_contype[g],m.geom_conaffinity[g], m.body_mass[m.geom_bodyid[g]])
