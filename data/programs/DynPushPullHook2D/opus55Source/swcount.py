import sys
from multiprocessing import Pool
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
def one(a):
    count,seed=a
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
        count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
    s, info = env.reset(seed=seed, options={'object_count': count})
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
    for t in range(300+150*count):
        s,rw,term,trunc,info=env.step(ap.get_action(s))
        if term: break
    return (seed, ap.n_sweeps, getattr(ap,"sweep_dir",None)), t+1, term
c=int(sys.argv[1])
with Pool(16) as p: res=p.map(one,[(c,s) for s in range(int(sys.argv[2]),int(sys.argv[3]))])
import collections
d=collections.defaultdict(list)
for ns,t,ok in res: d[ns].append(t)
for k in sorted(d): print(k, len(d[k]), sum(d[k])/len(d[k]))
for r in res: print(r[0], r[1], end="; ")
