import cProfile, pstats, sys, logging
logging.disable(logging.WARNING)
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
    count_kwarg="num_cubes", count_object_prefix="cube", constant_object_env_kwargs={'num_boxes': 1},
    design_counts=[1,2,3], eval_counts=[1,2,3,4])
k, seed = int(sys.argv[1]), int(sys.argv[2])
s, info = env.reset(seed=seed, options={'object_count': k})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap._sim(k)
cProfile.run('ap.plan = ap._make_plan(s)', 'prof.out')
p = pstats.Stats('prof.out'); p.sort_stats('tottime').print_stats(14)
