import sys
from env_client import make_env
from approach import GeneratedApproach, X_MIN, X_MAX
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
s0,s1=int(sys.argv[1]),int(sys.argv[2]); c=int(sys.argv[3]) if len(sys.argv)>3 else None
nb=nu=0
for seed in range(s0,s1):
    obs,info=env.reset(seed=seed, options=({'object_count':c} if c is not None else None))
    ap.reset(obs,info); ap._parse(obs)
    b=ap.block
    reach=lambda o: o["x"] + 0.006 <= X_MAX and o["x"] + o["w"] - 0.006 >= X_MIN
    if not reach(b): nb+=1; print('block unreachable', seed, b)
    d,g=ap._choose_block_dest(); ap.block_dest, ap.block_grel = d,g
    for o in ap.obst.values():
        if ap._column_hit(o) and not reach(o): nu+=1; print('unreach blocker', seed, o)
print('done', nb, nu)
