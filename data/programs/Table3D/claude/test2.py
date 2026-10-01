import numpy as np, sys, time
from env_client import make_env
from approach import GeneratedApproach
def run(seed, oc=None):
    env=make_env()
    try:
        obs,info=env.reset(seed=seed, options={"object_count":oc} if oc else None)
    except Exception as e:
        print("reset opts failed:",e); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    t0=time.time(); ap.reset(obs,info); steps=0; term=False
    for i in range(env.max_steps):
        a=ap.get_action(obs)
        obs,r,term,trunc,inf=env.step(a); steps+=1
        if term or trunc: break
    dt=time.time()-t0; env.close()
    return term, steps, dt, info.get("object_count")
if __name__=="__main__":
    oc=int(sys.argv[3]) if len(sys.argv)>3 else None
    ok=0;tot=0
    for s in range(int(sys.argv[1]),int(sys.argv[2])):
        term,steps,dt,c=run(s,oc); tot+=1; ok+=int(term)
        if not term or dt>10: print("seed",s,"count",c,"succ",term,"steps",steps,"t %.1f"%dt,flush=True)
    print("SUCCESS %d/%d"%(ok,tot))
