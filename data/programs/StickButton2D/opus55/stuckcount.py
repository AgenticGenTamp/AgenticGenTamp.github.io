import sys
from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
def run(seed):
    env=make_env()
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    st=0; idle=0
    for n in range(1000):
        a=ap.get_action(obs)
        if ap.stuck>0: st+=1
        if ap.idle>0: idle+=1
        obs,r,term,trunc,_=env.step(a)
        if term: break
    env.close()
    return seed, n+1, st, idle
with ThreadPoolExecutor(16) as ex:
    for r in ex.map(run, range(int(sys.argv[1]), int(sys.argv[2]))):
        if r[2] or r[3]: print(r)
