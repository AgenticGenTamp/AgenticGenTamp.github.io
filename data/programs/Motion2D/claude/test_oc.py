import sys, time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
oc=int(sys.argv[1]); n=int(sys.argv[2]); off=int(sys.argv[3]) if len(sys.argv)>3 else 0
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space,{})
fails=[]; st=[]; worst=0
for seed in range(off,off+n):
    obs,info=env.reset(seed=seed, options={"object_count":oc})
    t0=time.time(); ap.reset(obs,info); tot=0; term=False
    for i in range(env.max_steps):
        obs,r,term,tr,info=env.step(ap.get_action(obs)); tot+=1
        if term or tr: break
    el=time.time()-t0; worst=max(worst,el); st.append(tot)
    if not term: fails.append(seed)
print("oc",oc,"FAILS",fails,"mean",round(float(np.mean(st)),1),"worst_time",round(worst,1))
env.close()
