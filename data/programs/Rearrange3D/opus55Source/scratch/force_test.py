import numpy as np, mujoco, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env; 
for seed in (32,36):
  for F in (2,5,10,20,40):
    obs,_=env.reset(seed=seed)
    m, d = re.sim.model.mj_model, re.sim.data.mj_data
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
    act=np.zeros(11)
    for t in range(10): obs,*_=env.step(act)
    z0=obs[18]
    for t in range(30):
        d.xfrc_applied[bid,2]=F
        obs,*_=env.step(act)
    d.xfrc_applied[bid,:]=0
    for t in range(10): obs,*_=env.step(act)
    print(seed,"F",F,"mass",round(m.body_subtreemass[bid],3),"z0",round(z0,3),"zend",np.round(obs[16:19],3))
print("timestep",m.opt.timestep, "geom solref g142",m.geom_solref[142],m.geom_solimp[142],"friction",m.geom_friction[142], "cone",m.opt.cone,"noslip",m.opt.noslip_iterations)
for a in range(m.nu): print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a),m.actuator_forcerange[a],m.actuator_gainprm[a][:3],m.actuator_biasprm[a][:3],m.actuator_ctrlrange[a])
