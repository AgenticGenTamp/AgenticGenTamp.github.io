import cProfile, pstats, time, sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
s, info = env.reset(seed=46, options={'object_count': 3})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
m,target,surface,obst=ap._parse(s)
stats = {"travel":0, "fail":0, "checks":0}
ot = ap._travel
def tr(*a, **k):
    stats["travel"] += 1
    r = ot(*a, **k)
    if r is None: stats["fail"] += 1
    return r
ap._travel = tr
of = ap._fast_ok_wps
def fo(*a):
    stats["checks"] += 1
    return of(*a)
ap._fast_ok_wps = fo
t=time.time()
acts,preds,ok=ap._plan_mode(m,target,surface,obst,time.time()+30,sys.argv[1])
print(len(acts), time.time()-t, stats)
