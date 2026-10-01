import sys; sys.path.insert(0,'.')
import cProfile, pstats, time
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
def run():
    global obs
    for i in range(200):
        a=ap.get_action(obs); obs,r,term,trunc,_=env.step(a)
        if term or trunc: break
cProfile.run("run()", "scratch/prof.out")
pstats.Stats("scratch/prof.out").sort_stats("cumulative").print_stats(r"approach.py", 16)
