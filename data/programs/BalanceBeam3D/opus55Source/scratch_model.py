import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ke=env._kinder_env; oc=ke._object_centric_env; re=oc._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
obs,_=env.reset(seed=0)
print("njnt",m.njnt,"nq",m.nq)
for j in range(m.njnt): print(j, mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j), m.jnt_type[j], m.jnt_qposadr[j], m.jnt_range[j])
for s in range(m.nsite): print("site",s,mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,s), d.site_xpos[s])
print("base qpos", d.qpos[:3], "obs base", obs[16:19])
print("arm qpos obs", obs[19:27])
print("timestep", m.opt.timestep)
