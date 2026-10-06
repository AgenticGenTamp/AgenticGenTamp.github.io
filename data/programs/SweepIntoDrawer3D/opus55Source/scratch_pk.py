import sys, numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
s=int(sys.argv[1]); a0,a1,every=[int(x) for x in sys.argv[2].split(',')]; k=int(sys.argv[3])
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
print('cubes', np.array([obs[16*j:16*j+3] for j in range(5)]).round(3).tolist())
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for i in range(a1):
    a=ap.get_action(obs)
    if i>=a0 and i%every==0:
        print(i,'ee',ap.ee()[0].round(3),'cube',obs[16*k:16*k+3].round(3),'yaw',round(ap.cube_yaw(k),2),'grip',a[10],'fing',obs[146].round(3) if len(obs)>146 else '')
    obs,*_=env.step(a)
print(ap.log)
