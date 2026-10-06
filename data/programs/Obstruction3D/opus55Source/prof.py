import cProfile, pstats, sys
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
c=int(sys.argv[1]); seed=int(sys.argv[2])
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv", count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[c], eval_counts=[c])
s, info = env.reset(seed=seed, options={'object_count': c})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
def run():
    global s
    for i in range(800):
        s, r, term, trunc, _ = env.step(ap.get_action(s))
        if term: print("ok", i); break
cProfile.run('run()', 'prof.out')
p = pstats.Stats('prof.out'); p.sort_stats('cumulative').print_stats(25)
