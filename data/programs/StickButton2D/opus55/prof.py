import cProfile, pstats
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
obs,info=env.reset(seed=0, options={"object_count":5})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
ap._read(obs)
cProfile.run("ap._make_plan()", "prof.out")
pstats.Stats("prof.out").sort_stats("tottime").print_stats(12)
