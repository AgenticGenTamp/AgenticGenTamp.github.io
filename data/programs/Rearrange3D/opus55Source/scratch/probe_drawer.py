import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
seed=int(sys.argv[1]); drawers=sys.argv[2].split(',')
obs,_=env.reset(seed=seed); m,d=re.sim.model.mj_model,re.sim.data.mj_data
bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
dids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f"kitchen_island_drawer_{n}") for n in drawers]
jad=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"kitchen_island_drawer_{n}_joint")] for n in drawers]
act=np.zeros(11)
for t in range(40): obs,*_=env.step(act)
print("settled",np.round(obs[16:19],3))
for t in range(80):
    for b in dids: d.xfrc_applied[b,0]=30
    obs,*_=env.step(act)
for b in dids: d.xfrc_applied[b,0]=0
print("drawers",[round(d.qpos[j],3) for j in jad],"drink",np.round(obs[16:19],3))
for t in range(300):
    d.xfrc_applied[bid,0]=0.5
    obs,*_=env.step(act)
    if t%30==0: print(t,"drink",np.round(obs[16:23],3))
d.xfrc_applied[bid,:]=0
for t in range(40): obs,*_=env.step(act)
print("end drink",np.round(obs[16:23],3))
