import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
seed=int(sys.argv[1]); name=sys.argv[2]; F=float(sys.argv[3]); act=np.zeros(11)
obs,_=env.reset(seed=seed); m,d=re.sim.model.mj_model,re.sim.data.mj_data
b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f"kitchen_island_drawer_{name}")
j=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"kitchen_island_drawer_{name}_joint")]
for t in range(40): obs,*_=env.step(act)
for t in range(200):
    d.xfrc_applied[b,0]=F; obs,*_=env.step(act)
    if t%20==0: print(t,"q",round(d.qpos[j],3),"drink",np.round(obs[16:23],3),flush=True)
d.xfrc_applied[b,0]=0
for t in range(30): obs,*_=env.step(act)
print("end q",round(d.qpos[j],3),"drink",np.round(obs[16:23],3))
