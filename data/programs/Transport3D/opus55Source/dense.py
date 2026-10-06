import sys, logging, numpy as np, time
logging.disable(logging.WARNING)
import approach
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
k, seed = int(sys.argv[1]), int(sys.argv[2])
approach.PITCHES[:] = [tuple(map(float, sys.argv[3].split(',')))]
s, info = env.reset(seed=seed, options={'object_count': k})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.plan = ap._make_plan(s); 
sim = ap.sims[k]
fails=[l for l in sim.log if 'fail' in l]
placed=[l for l in sim.log if l.startswith('placed cube')]
print("pitch", sys.argv[3], "time %.1f"%(time.time()-t), "plan", len(ap.plan), "done", sim.done, "placed", len(placed), "fails", len(fails))
for l in fails[:8]: print(l)
names=[n for n in s.get_object_names() if n.startswith('cube') or n.startswith('box')]
for n in names:
    p=sim.obj_pos(n)
    if p[2]<0.3:
        others=sorted((np.linalg.norm(sim.obj_pos(m)[:2]-p[:2]),m) for m in names if m!=n)
        print("left", n, np.round(p,3), "nearest", [(round(d,3),m) for d,m in others[:2]])
print([l for l in sim.log if 'fail' in l and 'pick' not in l][:5])
