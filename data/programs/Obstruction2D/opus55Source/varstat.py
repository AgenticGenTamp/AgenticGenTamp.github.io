import sys, time, itertools
from multiprocessing import Pool
import numpy as np
def run(args):
    c, seed = args
    from robocode.environments.variable_object_count_env import VariableObjectCountEnv
    import approach, varstat2
    from approach import GeneratedApproach
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
    s,info=env.reset(seed=seed, options={'object_count':c})
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    ap._pick_cache = {}
    m,target,surface,obst=ap._parse(s)
    out=[]
    for v in approach.VARIANTS:
        t=time.time()
        acts,preds,ok=ap._plan_mode(m,target,surface,obst,time.time()+20,*v)
        out.append((len(acts) if ok else 999, time.time()-t))
    return c, seed, out
if __name__=="__main__":
    counts=[int(x) for x in sys.argv[1].split(',')]
    a,b=map(int,sys.argv[2].split('-'))
    with Pool(int(sys.argv[3])) as p:
        res=p.map(run,[(c,s) for c in counts for s in range(a,b)])
    import approach, varstat2
    V=approach.VARIANTS
    L=np.array([[o[0] for o in r[2]] for r in res]); T=np.array([[o[1] for o in r[2]] for r in res])
    for i,v in enumerate(V):
        print(v, "mean", L[:,i][L[:,i]<999].mean().round(2), "fails", (L[:,i]==999).sum(), "wins", (L[:,i]==L.min(1)).sum(), "tmax", T[:,i].max().round(1), "tmean", T[:,i].mean().round(2))
    print("min all", L.min(1).mean())
    print("min first3", L[:,:3].min(1).mean())
    for k in range(1,len(V)):
        best=None
        for sub in itertools.combinations(range(len(V)),k):
            v=L[:,list(sub)].min(1).mean()
            if best is None or v<best[0]: best=(v,sub)
        print(k, best)
    np.save("varstat.npy", L)
