import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
seed=int(sys.argv[1])
act=np.zeros(11)
for name in ("s1c0","s1c1"):
    obs,_=env.reset(seed=seed); m,d=re.sim.model.mj_model,re.sim.data.mj_data
    b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f"kitchen_island_drawer_{name}")
    j=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"kitchen_island_drawer_{name}_joint")]
    for t in range(40): obs,*_=env.step(act)
    for F in (5,30,100):
        for t in range(40): d.xfrc_applied[b,0]=F; obs,*_=env.step(act)
        print(seed,name,"F",F,"q",round(d.qpos[j],3),"drink",np.round(obs[16:19],3))
    d.xfrc_applied[b,0]=0
for F in (1,3,10):
  for dirn in ((1,0),(0,1),(0,-1)):
    obs,_=env.reset(seed=seed); m,d=re.sim.model.mj_model,re.sim.data.mj_data
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
    for t in range(40): obs,*_=env.step(act)
    p0=obs[16:19].copy()
    for t in range(60): d.xfrc_applied[bid,0:2]=np.array(dirn)*F; obs,*_=env.step(act)
    print("drinkF",F,dirn,np.round(p0,3),"->",np.round(obs[16:19],3))
print("dt per step",env._kinder_env if False else "", m.opt.timestep)
