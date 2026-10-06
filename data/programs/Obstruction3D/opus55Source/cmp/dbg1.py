import sys, time
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
import sys; sys.path.insert(0,"cmp"); from approach import GeneratedApproach, _Info
c=int(sys.argv[1]); seed=int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv", count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[c], eval_counts=[c])
s, info = env.reset(seed=seed, options={'object_count': c})
I=_Info(s)
for n in sorted(I.objs):
    if n!='robot': print('[dbg]', n, I.pos(n).round(3), I.he(n).round(3))
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); c0=time.process_time()
ap.reset(s, info)
for i in range(int(sys.argv[3]) if len(sys.argv)>3 else 1000):
    s, r, term, _, _ = env.step(ap.get_action(s))
    if term: print('[dbg] DONE', i, time.time()-t, 'cpu', time.process_time()-c0); break
    if time.time()-t > 70: print('[dbg] TIMEOUT'); break
