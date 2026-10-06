import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ke=env._kinder_env; oc=ke._object_centric_env; re=oc._robot_env
obs,_=env.reset(seed=0)
m=re.sim.model.mj_model; d=re.sim.data.mj_data
print(re.sim.data.mj_data is d)
obs,*_=env.step(np.zeros(11,dtype=np.float32))
print("base qpos", d.qpos[29:39], "\nobs", obs[16:26])
print("blk", d.qpos[22:25], obs[0:3])
