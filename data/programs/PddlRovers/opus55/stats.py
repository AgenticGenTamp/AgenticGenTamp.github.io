import sys
from env_client import make_env
from approach import GeneratedApproach
seeds=[int(s) for s in sys.argv[1].split(',')]
env=make_env(); agg={}
for seed in seeds:
    obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    hist={r:[] for r in ap.rnames}
    for t in range(200):
        a=ap.get_action(obs)
        pre={r:(obs.get(ap.rover_objs[r],'x'),obs.get(ap.rover_objs[r],'y'),obs.get(ap.rover_objs[r],'at_home')) for r in ap.rnames}
        obs,_,term,_,_=env.step(a)
        for i,r in enumerate(ap.rnames):
            x,y=obs.get(ap.rover_objs[r],'x'),obs.get(ap.rover_objs[r],'y')
            moved = abs(x-pre[r][0])+abs(y-pre[r][1])>1e-4
            cmd = abs(a[4*i])+abs(a[4*i+1])>1e-4
            hist[r].append((moved,cmd,round(float(a[4*i+3]),2),obs.get(ap.rover_objs[r],'at_home')))
        if term: break
    n=t+1
    out=[]
    for r in ap.rnames:
        h=hist[r]
        # last step where not (at home and stationary)
        last=max([i for i,(m,c,op,hm) in enumerate(h) if m or not hm]+[0])+1
        idle=sum(1 for (m,c,op,hm) in h[:last] if not m)
        rej=sum(1 for (m,c,op,hm) in h if c and not m)
        out.append((r,last,idle,rej))
    print(seed,n,out,flush=True)
