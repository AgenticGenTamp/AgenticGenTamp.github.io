import sys, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
s=int(sys.argv[1]); steps=[int(x) for x in sys.argv[2].split(',')]
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for i in range(max(steps)+1):
    if i in steps:
        print(i,'wiper',obs[147:154].round(3),'ee',ap.ee()[0].round(3))
        for g in range(m.ngeom):
            bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ''
            if 'wiper' in bn:
                c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
                R=d.geom_xmat[g].reshape(3,3); p=d.geom_xpos[g]+R@c; e=np.abs(R)@h
                print('   geom',(p-e).round(3),(p+e).round(3))
    a=ap.get_action(obs); obs,*_=env.step(a)
