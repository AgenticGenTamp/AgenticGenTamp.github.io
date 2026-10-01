import numpy as np, sys, os, time
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from approach import GeneratedApproach
def run(args):
    seed, cfg = args
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    for k,v in cfg.items(): setattr(ap,k,v)
    ap.reset(obs,info); t0=time.time()
    for t in range(env.max_steps):
        obs,r,te,tr,info=env.step(ap.get_action(obs))
        if te or tr: break
    env.close()
    nin=int((obs[2:80:16]<0.44).sum())
    return seed, cfg, te, t+1, nin, round(time.time()-t0,1)
if __name__=="__main__":
    cfgs=eval(sys.argv[1]); seeds=[int(s) for s in sys.argv[2:]]
    jobs=[(s,c) for c in cfgs for s in seeds]
    with ThreadPoolExecutor(int(os.environ.get("NJ","12"))) as ex:
        for r in ex.map(run, jobs): print(r, flush=True)
