import sys; sys.path.insert(0,'.')
import numpy as np, time
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 60
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info); ap.debug=True
cfg,blocks,*_=ap._parse(obs)
for n,b in blocks.items(): print(n,np.round(b,3))
for i in range(N):
    t=time.time(); a=ap.get_action(obs); dt=time.time()-t
    obs,r,term,tr,_=env.step(a)
    cfg,blocks,holding,held,*_=ap._parse(obs)
    print(i+1, "dt%.2f"%dt, "plan",len(ap.plan),"tgt",ap.target,"hold",held,"base",cfg[:3].round(2),"a",a.round(2)[[0,1,2,10]], "fail",ap.fail)
    if i in (9,10): print({n:np.round(b,3) for n,b in blocks.items()}, ap._on_plate(blocks["block0"]))
    if term: print("DONE"); break
