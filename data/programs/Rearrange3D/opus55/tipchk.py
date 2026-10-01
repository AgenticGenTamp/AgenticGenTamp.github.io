import sys, numpy as np, approach
from env_client import make_env
from approach import GeneratedApproach
from kin import fk_world
np.set_printoptions(precision=3,suppress=True,linewidth=200)
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
for t in range(int(sys.argv[2])):
    a=ap.get_action(obs)
    if t>=int(sys.argv[3]):
        p,R=fk_world(obs[93:96],obs[96:103]); o=int(sys.argv[4])
        print(t,'tip',p,'yaw %.2f'%np.arctan2(R[1,0],R[0,0]),'obj',obs[o:o+3],'grip',a[10], 'fing',obs[103:105])
    obs,r,te,tr,info=env.step(a)
