import sys, numpy as np, mujoco
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
s=int(sys.argv[1]); a0,a1=int(sys.argv[2]),int(sys.argv[3])
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
re=env._kinder_env._object_centric_env._robot_env
m=re.sim.model.mj_model; d=re.sim.data.mj_data
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
def nm(g):
    b=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]); return b
for i in range(a1):
    a=ap.get_action(obs); obs,*_=env.step(a)
    if i>=a0:
        cs=set()
        for c in d.contact[:d.ncon]:
            n1,n2=nm(c.geom1),nm(c.geom2)
            if 'robot' in n1 or 'robot' in n2: cs.add((n1,n2))
        print(i, ap.ee()[0].round(3), sorted(cs))
