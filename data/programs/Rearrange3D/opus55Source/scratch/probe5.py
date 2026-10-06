import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
seed=int(sys.argv[1]); act=np.zeros(11)
for F in [float(x) for x in sys.argv[2].split(',')]:
    obs,_=env.reset(seed=seed); m,d=re.sim.model.mj_model,re.sim.data.mj_data
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
    for t in range(40): obs,*_=env.step(act)
    zmax=0
    for t in range(40):
        d.xfrc_applied[bid,2]=F*min(1,(t+1)/20); obs,*_=env.step(act); zmax=max(zmax,obs[18])
        if obs[18]>0.48: break
    d.xfrc_applied[bid,:]=0
    for t in range(40): obs,*_=env.step(act)
    print(seed,"F",F,"zmax",round(zmax,3),"end",np.round(obs[16:23],3),flush=True)
