import cProfile, pstats
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",count_kwarg="num_obstructions",count_object_prefix="obstruction",bilevel_env_name="obstruction2d",design_counts=[0,1,2,3,4],eval_counts=[0,1,2,3,4,5])
s, info = env.reset(seed=46, options={"object_count": 3})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
cProfile.run("ap.reset(s, info)", "prof.out")
p = pstats.Stats("prof.out"); p.sort_stats("tottime").print_stats(12)
