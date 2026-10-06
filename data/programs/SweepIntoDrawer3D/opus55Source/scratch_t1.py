import time, numpy as np, mujoco
t=time.time()
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
print("create+reset", time.time()-t)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
print(m.njnt, m.nbody)
for j in range(m.njnt):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
    if n and ('robot' in n or 'drawer_s1c1' in n or 'joint' in n): print(j,n,m.jnt_qposadr[j],m.jnt_dofadr[j],m.jnt_range[j])
# geoms of robot base
for g in range(m.ngeom):
    b=m.geom_bodyid[g]; bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)
    if bn and ('base' in bn.lower()) and m.geom_contype[g]+m.geom_conaffinity[g]>0:
        print('geom',g,bn,m.geom_type[g],m.geom_size[g],d.geom_xpos[g].round(3))
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
print('pinch',d.site_xpos[sid], d.site_xmat[sid].reshape(3,3).round(3))
