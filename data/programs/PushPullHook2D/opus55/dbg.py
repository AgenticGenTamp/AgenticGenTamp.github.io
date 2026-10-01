import sys, numpy as np
from env_client import make_env
import approach
from approach import GeneratedApproach
np.set_printoptions(precision=3, suppress=True, linewidth=200)
seed=int(sys.argv[1]); lo,hi=(int(sys.argv[3]),int(sys.argv[4])) if len(sys.argv)>4 else (-1,-1)
env = make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
print('init', obs[:3], 'hook', obs[9:12], 'B', obs[20:22], 'T', obs[29:31])
for t in range(int(sys.argv[2]) if len(sys.argv)>2 else 300):
    a=ap.get_action(obs); obs,r,te,tr,_=env.step(a)
    if t%10==0 or te or (lo<=t<=hi): print(t, 'a',a, 'r',obs[:3],'vac',obs[6],'hook',obs[9:12],'B',obs[20:22], 'd', np.linalg.norm(obs[20:22]-obs[29:31]))
    if te: break
