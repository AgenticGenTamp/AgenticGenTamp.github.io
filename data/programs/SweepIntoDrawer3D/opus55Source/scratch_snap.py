import sys, numpy as np, time
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
import imageio
s=int(sys.argv[1]); steps=[int(x) for x in sys.argv[2].split(',')]
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0", render_mode="rgb_array") if False else KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for i in range(max(steps)+1):
    if i in steps:
        img=env.render()
        imageio.imwrite(f'snap_{s}_{i}.png', img)
        print(i,'cubes',np.array([obs[16*k:16*k+3] for k in range(5)]).round(3).tolist())
    a=ap.get_action(obs); obs,*_=env.step(a)
