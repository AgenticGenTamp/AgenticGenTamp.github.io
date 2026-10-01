import numpy as np, sys
from multiprocessing import Pool
def run(rng):
    from env_client import make_env
    import approach, importlib
    env=make_env(); ap=approach.GeneratedApproach(env.action_space,env.observation_space,{})
    out=[]
    for s in rng:
        o,i=env.reset(seed=s); ap.reset(o,i); tgt=o[19:22].copy()
        for t in range(env.max_steps):
            o,r,te,tr,_=env.step(ap.get_action(o))
            if te or tr: break
        out.append((s,te,t+1,tgt.tolist(),o[:3].tolist()))
    env.close(); return out
if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2]); W=8
    chunks=[list(range(a+k,b,W)) for k in range(W)]
    with Pool(W) as p: res=sum(p.map(run,chunks),[])
    res.sort()
    fails=[r for r in res if not r[1]]
    for f in fails: print('FAIL',f[0],np.round(f[3],3),np.round(f[4],3))
    print('n',len(res),'fails',len(fails),'mean len',np.mean([r[2] for r in res]))
    import approach as A
    ex=[]
    for r in res:
        if r[1]:
            g=A.plan_goal(np.zeros(2),np.array(r[3][:2])); lb=int(np.ceil(np.max(np.abs(g))/0.4-1e-9))
            if r[2]>lb: ex.append((r[0],r[2],lb))
    print('excess',len(ex),ex[:20])
