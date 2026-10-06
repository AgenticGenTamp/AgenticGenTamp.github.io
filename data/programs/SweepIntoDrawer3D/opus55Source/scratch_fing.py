import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
for _ in range(3): obs,*_=env.step(np.zeros(11))
sp=d.site_xpos[sid]; sR=d.site_xmat[sid].reshape(3,3)
for g in range(m.ngeom):
    if m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
    p=sR.T@(d.geom_xpos[g]-sp)
    if np.linalg.norm(p)<0.2:
        R=sR.T@d.geom_xmat[g].reshape(3,3); h=np.abs(R)@m.geom_aabb[g][3:]; c=p+R@m.geom_aabb[g][:3]
        print(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g), mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]), 'lo',(c-h).round(3),'hi',(c+h).round(3))
