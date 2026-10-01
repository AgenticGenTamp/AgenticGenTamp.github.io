import sys, numpy as np, json
from multiprocessing import Pool
def run(args):
    lo,hi=args
    from env_client import make_env
    from approach import GeneratedApproach
    env=make_env(); out=[]
    for s in range(lo,hi):
        obs,info=env.reset(seed=s)
        ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
        term=False
        for i in range(env.max_steps):
            obs,r,term,trunc,inf=env.step(ap.get_action(obs))
            if term or trunc: break
        out.append((s,i+1,bool(term)))
    env.close(); return out
if __name__=="__main__":
    lo,hi,nw=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)>3 else 8
    chunks=[]; step=(hi-lo)//nw+1
    for a in range(lo,hi,step): chunks.append((a,min(a+step,hi)))
    with Pool(len(chunks)) as p: res=sum(p.map(run,chunks),[])
    fails=[s for s,st,t in res if not t]
    steps=[st for s,st,t in res if t]
    print("n",len(res),"solved",len(steps),"mean steps",np.mean(steps),"max",max(steps),"fails",fails)
    worst=sorted([(st,s) for s,st,t in res if t])[-10:]
    print("worst:",worst)
    json.dump(res,open("sweep_%d_%d.json"%(lo,hi),"w"))
