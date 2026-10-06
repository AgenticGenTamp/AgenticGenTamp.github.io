import sys
from multiprocessing import Pool
def run(args):
    n,seed=args
    import time, numpy as np
    from robocode.environments.variable_object_count_env import VariableObjectCountEnv
    from approach import GeneratedApproach
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic3d.task_families:Shelf3DEnv",count_kwarg="num_objects",count_object_prefix="cube",constant_object_env_kwargs={'scene_bg': False},design_counts=[n],eval_counts=[n])
    s,info=env.reset(seed=seed,options={'object_count':n})
    ap=GeneratedApproach(env.action_space,env.observation_space,{})
    t=time.time(); ap.reset(s,info)
    maxsteps=300+150*n; done=False
    for i in range(maxsteps):
        s,r,term,trunc,info=env.step(ap.get_action(s))
        if term: done=True;break
    ing=sum(ap._in_goal(ap._cube(s,c)) for c in ap.cubes)
    return (n,seed,done,i+1,ing,round(time.time()-t,1))
if __name__=="__main__":
    counts=[int(x) for x in sys.argv[1].split(',')]
    s0,s1=int(sys.argv[2]),int(sys.argv[3])
    jobs=[(n,s) for n in counts for s in range(s0,s1)]
    with Pool(int(sys.argv[4]) if len(sys.argv)>4 else 8) as p:
        res=p.map(run,jobs,chunksize=1)
    for r in res:
        if not r[2]: print("FAIL",r)
    import numpy as np
    for n in counts:
        st=[r[3] for r in res if r[0]==n and r[2]]
        print("n",n,"mean steps",np.mean(st).round(1),"max",max(st))
    for r in sorted(res,key=lambda r:-r[3]/(300+150*r[0]))[:6]: print("slow",r)
    print("success",sum(r[2] for r in res),"/",len(res), "max steps frac", max(r[3]/(300+150*r[0]) for r in res if r[2]), "max time", max(r[5] for r in res))
