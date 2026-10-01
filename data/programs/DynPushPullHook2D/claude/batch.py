import sys, numpy as np, multiprocessing as mp
def run(seeds):
    from env_client import make_env
    from approach import GeneratedApproach
    env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
    out=[]
    for seed in seeds:
        obs,info=env.reset(seed=seed); ap.reset(obs,info); term=False; n=0
        for i in range(env.max_steps):
            obs,r,term,tr,info=env.step(ap.get_action(obs)); n+=1
            if term or tr: break
        out.append((seed, info.get('object_count'), bool(term), n, ap.phase))
    env.close(); return out
if __name__=='__main__':
    a,b,w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 8
    seeds=list(range(a,b)); chunks=[seeds[i::w] for i in range(w)]
    with mp.Pool(w) as p: res=sum(p.map(run,chunks),[])
    res.sort()
    fails=[r for r in res if not r[2]]
    for r in res:
        if not r[2]: print("FAIL",r)
    ok=[r for r in res if r[2]]
    print("solved",len(ok),"/",len(res),"mean steps",round(np.mean([r[3] for r in ok]),1) if ok else 0)
