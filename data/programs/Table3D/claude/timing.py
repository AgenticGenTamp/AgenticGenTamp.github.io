import numpy as np, time, sys
from env_client import make_env
from approach import GeneratedApproach
oc=int(sys.argv[1]) if sys.argv[1]!="none" else None
ok=0;tot=0;worst=[]
for s in range(500,520):
    env=make_env(); obs,info=env.reset(seed=s, options=({"object_count":oc} if oc else None))
    ap=GeneratedApproach(env.action_space,env.observation_space,{})
    t0=time.time(); ap.reset(obs,info); term=False; steps=0
    for i in range(env.max_steps):
        a=ap.get_action(obs); obs,r,term,tr,_=env.step(a); steps+=1
        if term or tr: break
    dt=time.time()-t0; env.close(); tot+=1; ok+=int(term); worst.append((round(dt,2),steps,s))
    if not term: print("FAIL seed",s,"oc",oc,flush=True)
worst.sort(reverse=True)
print("oc",oc,"success %d/%d"%(ok,tot),"slowest",worst[:3])
