import sys, numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
count=int(sys.argv[1]); seed=int(sys.argv[2]); every=int(sys.argv[3]) if len(sys.argv)>3 else 4
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
s, info = env.reset(seed=seed, options={'object_count': count})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
for t in range(300+150*count):
    a=ap.get_action(s)
    if ap.phase=="moves" and t%every==0:
        r,h,tb,ob=ap._read(s)
        C=ap._corner(r)
        print(t, ap.moves[0][0] if ap.moves else "-", ap.moves[0][1] if ap.moves and ap.moves[0][1] else "", "C",np.round(C,2),"cmd",np.round(ap.cmd,2),"th",round(r["th"],2),"R",round(r["x"],2),round(r["y"],2),round(r["arm"],2),"T",round(tb["x"],2),round(tb["y"],2), "nm",len(ap.moves))
    s,rw,term,trunc,info=env.step(a)
    if term: break
print("steps",t+1)
