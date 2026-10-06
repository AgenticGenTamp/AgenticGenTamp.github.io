import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
obs,_=env.reset(seed=0)
re = env._kinder_env._object_centric_env._robot_env; m, d = re.sim.model.mj_model, re.sim.data.mj_data
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
print("site xmat\n", d.site_xmat[sid].reshape(3,3).round(3), d.site_xpos[sid].round(3))
for i in range(m.nbody):
    n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i)
    if n and ('pad' in n or 'finger' in n or 'follower' in n): print(n, d.xpos[i].round(3))
for j in ['robot_joint_x','robot_joint_y','robot_joint_th']:
    print(j, d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j)]])
print(obs[93:104])
print("nq",m.nq, "joint names", [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)][:50])
