import sys, numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
s=int(sys.argv[1])
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
prev=None
for i in range(int(sys.argv[2])):
    a=ap.get_action(obs)
    d=round(float(obs[107]),3)
    if prev is None or abs(d-prev)>0.003 or i%25==0:
        print(i,'dr',d,'ee',ap.ee()[0].round(3),'base',obs[125:128].round(3),'grip',a[10],'cz',np.array([obs[16*k+2] for k in range(5)]).round(3))
        prev=d
    obs,*_=env.step(a)
