import time, cProfile, pstats
from robocode.environments.rovers_variable_count_env import RoversVariableCountEnv
from approach import GeneratedApproach
env = RoversVariableCountEnv(design_counts=[1,2,3,4], eval_counts=[1,2,3,4])
state, info = env.reset(seed=0)
a = GeneratedApproach(env.action_space, env.observation_space, {})
cProfile.run("a.reset(state, info)", "/tmp/prof")
pstats.Stats("/tmp/prof").sort_stats("tottime").print_stats(12)
