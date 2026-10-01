import numpy as np, sys, json
from multiprocessing import Pool
from env_client import make_env
from approach import GeneratedApproach

def run(args):
    seed, cfg = args
    env=make_env(); o,i=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space,env.observation_space,{})
    for k,v in cfg.items():
        setattr(ap,k,v)
    ap.reset(o,i)
    for k,v in cfg.items():
        setattr(ap,k,v)
    if 'ellmul' in cfg:
        ap.ell = ap.ell0 = 0.5*ap.Lh*cfg['ellmul']
    if 'C2K' in cfg:
        ap.c2 = cfg['C2K']*ap.Lh*ap.Lv
    te=False
    for k in range(env.max_steps):
        a=np.clip(np.asarray(ap.get_action(o),float),-0.0499,0.0499)
        o,r,te,tr,_=env.step(a)
        if te or tr: break
    env.close()
    return (seed, bool(te), k+1)

if __name__=="__main__":
    cfgs=json.loads(sys.argv[1])
    n=int(sys.argv[2]) if len(sys.argv)>2 else 60
    off=int(sys.argv[3]) if len(sys.argv)>3 else 0
    seeds=list(range(off,off+n))
    if len(sys.argv)>4: seeds=[int(x) for x in sys.argv[4].split(',')]
    n=len(seeds)
    for name,cfg in cfgs.items():
        with Pool(12) as p:
            res=p.map(run, [(s,cfg) for s in seeds])
        ok=sum(1 for r in res if r[1])
        steps=[r[2] for r in res]
        fails=[r[0] for r in res if not r[1]]
        print(name, "solved",ok,"/",n,"avg",round(float(np.mean(steps)),1),"fails",fails)
