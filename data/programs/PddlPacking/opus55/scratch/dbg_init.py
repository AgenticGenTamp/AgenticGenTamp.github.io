import sys; sys.path.insert(0,'.')
import numpy as np, math
from env_client import make_env
from approach import *
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
cfg,blocks,holding,held,gtf,gq=ap._parse(obs)
w=World(blocks)
for n,b in blocks.items():
    print(n, np.round(b,3))
    print(" baselist", [(lb,np.round(bb,2).tolist()) for lb,bb in ap._base_list(cfg,b[0],b[1],False)[:8]])
    for sc,gc in ap._grasp_options(cfg,n,b,w)[:6]:
        bp=ap._best_path(cfg,gc,w,closed=False)
        print(" ",sc, gc[:3].round(2), "path", None if bp is None else (len(bp[0]),bp[1]))
