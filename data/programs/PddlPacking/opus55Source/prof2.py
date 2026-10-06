import cProfile, pstats, sys
import numpy as np
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv
from approach import GeneratedApproach
env = PR2PackedVariableCountEnv([9],[9])
obs, info = env.reset(seed=19, options={'object_count': 9})
ap = GeneratedApproach(env.action_space, env.observation_space, None)
ap.reset(obs, info)
pr = cProfile.Profile(); pr.enable()
for i in range(3):
    a = ap.get_action(obs); obs, r, te, tr, _ = env.step(a)
pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(25)
