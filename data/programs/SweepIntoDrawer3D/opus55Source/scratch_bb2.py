import sys, numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
s=int(sys.argv[1]); a0,a1,every=[int(x) for x in sys.argv[2].split(',')]
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for i in range(a1):
    a=ap.get_action(obs)
    if i>=a0 and i%every==0:
        print(i,'ee',ap.ee()[0].round(3),'bb',round(ap.blade_bottom(),3),'w',obs[147:154].round(3),'grip',a[10], 'base',obs[125:128].round(3))
    obs,*_=env.step(a)
