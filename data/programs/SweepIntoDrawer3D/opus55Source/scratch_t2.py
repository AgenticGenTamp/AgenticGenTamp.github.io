import time, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
# robot geoms world aabb
lo=np.full(3,9.);hi=-lo
for g in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
    if bn.startswith('robot_base') and m.geom_contype[g]+m.geom_conaffinity[g]>0:
        c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
        R=d.geom_xmat[g].reshape(3,3); p=d.geom_xpos[g]+R@c; e=np.abs(R)@h
        print(bn, (p-e).round(3),(p+e).round(3))
print('base',obs[125:128])
# fingertip geoms
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'robot_pinch_site')
for g in range(m.ngeom):
    bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
    if ('pad' in bn or 'follower' in bn or 'finger' in bn) and m.geom_contype[g]+m.geom_conaffinity[g]>0:
        R=d.site_xmat[sid].reshape(3,3)
        print(bn, m.geom_type[g], m.geom_size[g].round(3), (R.T@(d.geom_xpos[g]-d.site_xpos[sid])).round(3))
# arm tracking test
q0=obs[128:135].copy()
for i in range(15):
    a=np.zeros(11); a[3]=0.1; a[4]=0.1
    obs,*_=env.step(a)
    print(i, (obs[128:130]-q0[:2]).round(3), obs[139:141].round(3))
t=time.time()
for i in range(10): env.step(np.zeros(11))
print('t/step',(time.time()-t)/10)
