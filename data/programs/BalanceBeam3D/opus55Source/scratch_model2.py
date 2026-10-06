import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ke=env._kinder_env; oc=ke._object_centric_env; re=oc._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
obs,_=env.reset(seed=0)
mujoco.mj_forward(m,d)
print("base qpos", d.qpos[29:32], "obs", obs[16:19])
print("site", d.site_xpos[4], "\n", d.site_xmat[4].reshape(3,3))
for b in range(m.nbody):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)
    if n and n.startswith('robot'): print(b,n,np.round(d.xpos[b],3))
# base geoms
for g in range(m.ngeom):
    b=m.geom_bodyid[g]; bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)
    if bn in ('robot_base_link','robot_base') or (bn and 'base' in bn and 'robot' in bn):
        print('geom',g,bn,m.geom_type[g],np.round(m.geom_size[g],3),np.round(d.geom_xpos[g],3), m.geom_contype[g], m.geom_conaffinity[g])
print("ctrlrange", m.actuator_ctrlrange)
