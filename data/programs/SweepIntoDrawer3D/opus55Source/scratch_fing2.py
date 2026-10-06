import numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
names=['robot_left_pad1','robot_left_follower']
gids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'robot_left_pad1')]
fb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot_left_follower')
fg=[g for g in range(m.ngeom) if m.geom_bodyid[g]==fb]
for c in [0,0.2,0.3,0.4,0.5,0.6,0.7]:
    a=np.zeros(11); a[10]=c
    for _ in range(8): obs,*_=env.step(a)
    sp=d.site_xpos[sid]; sR=d.site_xmat[sid].reshape(3,3)
    out=[]
    for g in gids+fg:
        R=sR.T@d.geom_xmat[g].reshape(3,3); p=sR.T@(d.geom_xpos[g]-sp); h=np.abs(R)@m.geom_aabb[g][3:]; cc=p+R@m.geom_aabb[g][:3]
        out.append(((cc-h)[0].round(3),(cc+h)[0].round(3)))
    print(c,out, 'obs146',obs[146].round(3))
