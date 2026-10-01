import cProfile, pstats, sys
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]), options={'object_count':8})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
pr=cProfile.Profile(); pr.enable(); ap.reset(obs, info); pr.disable()
print('layered', ap.layered, ap.margin)
pstats.Stats(pr).sort_stats('tottime').print_stats(10)
