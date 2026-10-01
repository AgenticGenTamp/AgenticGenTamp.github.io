import sys; sys.path.insert(0,'.')
import cProfile, pstats, time
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
t=time.time()
cProfile.run("a=ap.get_action(obs)", "scratch/prof.out")
print("time", time.time()-t, a, len(ap.plan))
pstats.Stats("scratch/prof.out").sort_stats("cumulative").print_stats(15)
