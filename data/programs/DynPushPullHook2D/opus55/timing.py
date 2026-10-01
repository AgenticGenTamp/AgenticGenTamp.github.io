import sys,time,os
from env_client import make_env
from approach import GeneratedApproach
e=make_env()
for s in range(int(sys.argv[1]),int(sys.argv[2])):
    o,i=e.reset(seed=s,options={'object_count':int(os.environ.get('OC',3))})
    ap=GeneratedApproach(e.action_space,e.observation_space,{}); ap.reset(o,i)
    tt=0;mx=0
    for t in range(1000):
        t0=time.time(); a=ap.get_action(o); dt=time.time()-t0; tt+=dt; mx=max(mx,dt)
        o,r,term,tr,_=e.step(a)
        if term or tr: break
    print(s,term,t+1,'policy time %.2f max step %.3f'%(tt,mx))
