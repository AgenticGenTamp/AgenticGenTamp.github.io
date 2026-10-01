import sys, time, json
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
def run(seed, oc=None, max_steps=1000):
    env=make_env()
    kw={} if oc is None else {'options':{'object_count':oc}}
    obs,info=env.reset(seed=seed,**kw)
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs,info)
    t0=time.time()
    for i in range(max_steps):
        a=ap.get_action(obs)
        obs,r,term,trunc,info=env.step(a)
        if term:
            env.close(); return True,i+1,time.time()-t0,ap.stage
        if trunc: break
    st=ap.stage; env.close(); return False,max_steps,time.time()-t0,st
if __name__=="__main__":
    oc = None if sys.argv[1]=='n' else int(sys.argv[1])
    seeds=[int(s) for s in sys.argv[2:]]
    ok=0; tot=0; worst=0
    for s in seeds:
        try:
            r=run(s,oc)
        except Exception as e:
            print(f"seed {s} ERROR {repr(e)[:120]}", flush=True); continue
        ok+=r[0]; tot+=r[1]; worst=max(worst,r[2])
        print(f"seed {s} oc={oc}: {'OK ' if r[0] else 'FAIL'} steps={r[1]} wall={r[2]:.1f} stage={r[3]}", flush=True)
    print(f"== success {ok}/{len(seeds)} avg_steps={tot/max(1,len(seeds)):.0f} worst_wall={worst:.1f}")
