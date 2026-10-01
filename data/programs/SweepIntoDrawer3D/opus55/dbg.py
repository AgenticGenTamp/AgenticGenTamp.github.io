import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
from kin import fk_world
np.set_printoptions(precision=3, suppress=True, linewidth=200)
seed=int(sys.argv[1]); N=int(sys.argv[2]); every=int(sys.argv[3]) if len(sys.argv)>3 else 5
env=make_env(); obs,info=env.reset(seed=seed)
import os
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.mode=os.environ.get("MODE","sweep"); ap.pgrip=float(os.environ.get("PG","1")); ap.reset(obs,info)
for t in range(N):
    a=ap.get_action(obs); obs,r,te,tr,info=env.step(a)
    if t%every==0:
        C=obs[:80].reshape(5,16)[:,:3]
        print(t, "tool",fk_world(obs[125:128],obs[128:135])[:3,3], "base",obs[125:128],"dr %.3f"%obs[107], "g%.0f"%obs[135], "r",r, "Cz",C[:,2].round(2), "Cxy", C[:,:2].round(2).ravel(), "W", obs[147:154].round(3))
    if te: print("TERMINATED", t); break
np.save('dbg_obs.npy', obs)
env.close()
