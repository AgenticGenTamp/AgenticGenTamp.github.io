import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
W=obs.get_object_from_name('wiper_0'); R=obs.get_object_from_name('robot')
last=None
for t in range(int(sys.argv[2])):
    a = ap.get_action(obs)
    if ap.phase in ('close','lift') :
        c=ap.cubes[ap.target]
        print(t,ap.phase,ap.target,'cube',c[:3].round(3),'wiper',[round(obs.get(W,f),2) for f in ['x','y','z','qw','qx','qy','qz']],'grip',round(obs.get(R,'pos_gripper'),2),round(obs.get(R,'vel_gripper'),3))
    obs, r, term, trunc, info = env.step(a)
