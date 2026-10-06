import sys, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
s=int(sys.argv[1]); a0,a1,every=[int(x) for x in sys.argv[2].split(',')]
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
blade=[g for g in range(m.ngeom) if 'wiper' in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or '')]
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for i in range(a1):
    if i>=a0 and i%every==0:
        out=[]
        for g in blade:
            c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
            R=d.geom_xmat[g].reshape(3,3); p=d.geom_xpos[g]+R@c; e=np.abs(R)@h
            out.append(np.r_[p-e,p+e].round(3))
        cub=np.array([obs[16*k:16*k+3] for k in range(5)]).round(3)
        print(i,'ee',ap.ee()[0].round(3),'blade',out[1],'cubes x',cub[:,0],'z',cub[:,2], 'y', cub[:,1])
    a=ap.get_action(obs); obs,*_=env.step(a)
