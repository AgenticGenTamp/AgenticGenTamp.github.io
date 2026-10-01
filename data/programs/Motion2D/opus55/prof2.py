import time, cProfile, pstats
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0, options={'object_count':8})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.reset(obs, info); print('plan', time.time()-t, ap.margin, round(ap.ptheta,2))
pr=cProfile.Profile(); pr.enable()
for i in range(200):
    obs,r,term,*_=env.step(ap.get_action(obs))
    if term: break
pr.disable(); pstats.Stats(pr).sort_stats('cumtime').print_stats(8)
