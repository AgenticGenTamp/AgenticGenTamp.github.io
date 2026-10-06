import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
seed=int(sys.argv[1]); act=np.zeros(11)
for F in [(10,0,0),(-10,0,0),(0,10,0),(0,-10,0),(30,0,0),(0,30,0),(0,-30,0),(15,0,15)]:
    obs,_=env.reset(seed=seed); m,d=re.sim.model.mj_model,re.sim.data.mj_data
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
    for t in range(40): obs,*_=env.step(act)
    p0=obs[16:19].copy()
    for t in range(60): d.xfrc_applied[bid,0:3]=F; obs,*_=env.step(act)
    d.xfrc_applied[bid,:]=0
    for t in range(10): obs,*_=env.step(act)
    print(seed,"F",F,np.round(p0,3),"->",np.round(obs[16:19],3),flush=True)
