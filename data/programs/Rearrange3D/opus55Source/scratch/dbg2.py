import sys, numpy as np
sys.path.insert(0,'.')
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=int(sys.argv[1])); ap.reset(obs,info)
for k in range(int(sys.argv[2])):
    a=ap.get_action(obs)
    if ap.phase==sys.argv[3] and k%4==0:
        print(k, ap.seg['k'], ap.seg['n'], "qerr",(ap.q_target-obs[96:103]).round(3), "vel", np.abs(obs[107:114]).max().round(3))
    obs,*_=env.step(a)
