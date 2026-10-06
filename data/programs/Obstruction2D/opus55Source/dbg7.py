import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
c, seed = int(sys.argv[1]), int(sys.argv[2])
s,info=env.reset(seed=seed, options={'object_count':c})
ap=GeneratedApproach(env.action_space, env.observation_space, {})
m,target,surface,obst=ap._parse(s)
for mode in ["float","park","cost","clear"]:
    t=time.time()
    acts,preds,ok=ap._plan_mode(m,target,surface,obst,time.time()+30,mode)
    print(mode, ok, len(acts), round(time.time()-t,1))
orig = ap._try_move
def tm(m, name, *a, **k):
    r = orig(m, name, *a, **k)
    print("  move", name, "from", [round(v,3) for v in m.blocks[name][:2]], "robot", round(m.rx,3), "->", None if r is None else ([round(v,3) for v in r[2].blocks[name][:2]], len(r[0])))
    return r
ap._try_move = tm
ot = ap._try_target
def tt(m, *a):
    r = ot(m, *a); print("  target", None if r is None else len(r[0])); return r
ap._try_target = tt
print(surface)
ap._plan_mode(m,target,surface,obst,time.time()+30,"park")
