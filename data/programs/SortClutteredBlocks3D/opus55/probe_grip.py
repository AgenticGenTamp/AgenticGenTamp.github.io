# test: grasp success vs pre-open level and lateral offset along finger axis
import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach, GRASP_DZ
import approach
res=[]
for gopen in [0.3,0.5,0.6]:
  for off in [0.0, 0.01, 0.02]:
    env = make_env(); obs, info = env.reset(seed=5)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    # force target: isolated cube -> pick highest-x cube
    ap.phase='goto_pick'; 
    ap.target = max(ap.cube_names, key=lambda n: ap.cubes[n]['p'][0])
    orig = ap.cubes[ap.target]['p'].copy()
    ok=None
    for t in range(150):
        a = ap.get_action(obs)
        if ap.phase=='goto_pick' or ap.phase=='descend':
            a[10]=0.0 if t<3 else gopen
            # shift base along y (finger axis) by off
            pass
        obs, r, term, trunc, info = env.step(a)
        if ap.phase in ('transport','release','select') :
            ok = ap.phase!='select' ; break
    print(gopen, off, ok, flush=True)
    env.close()
    break
