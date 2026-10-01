import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); T=int(sys.argv[2]); cnt=int(sys.argv[3]) if len(sys.argv)>3 else 4
env=make_env(); obs,info=env.reset(seed=seed,options={"object_count":cnt})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T):
    obs,r,term,trunc,info=env.step(ap.get_action(obs))
    if term: break
ap._parse(obs)
for n,c in sorted(ap.cubes.items()):
    print(n, ap._cube_color(n), np.round(c["p"],3), ap._which_bin(c["p"]), np.round(c["quat"],2))
print("pair", ap._pair_plan(), "sel", ap._select(), ap.sel_clr, ap.fail)
