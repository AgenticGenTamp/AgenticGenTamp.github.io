import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
obs,_=env.reset(seed=32)
re = env._kinder_env._object_centric_env._robot_env; m, d = re.sim.model.mj_model, re.sim.data.mj_data
for g in range(m.ngeom):
    if m.geom_type[g]!=6: continue
    p=d.geom_xpos[g]; s=m.geom_size[g]; R=d.geom_xmat[g].reshape(3,3)
    ext=np.abs(R)@s
    if 0.40<p[2]+ext[2]<0.50 and m.geom_contype[g]:
        print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g), mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]),"x",np.round([p[0]-ext[0],p[0]+ext[0]],3),"y",np.round([p[1]-ext[1],p[1]+ext[1]],3),"z",np.round([p[2]-ext[2],p[2]+ext[2]],3))
print("bowl",obs[0:3],"drink",obs[16:19],"can",obs[32:35],"base",obs[93:96])
