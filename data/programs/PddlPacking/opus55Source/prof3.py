import cProfile, pstats, sys, time
import numpy as np
from approach import GeneratedApproach, obs_to_flat
from robocode.environments.pr2_tamp_variable_count_env import PR2PackedVariableCountEnv
k=int(sys.argv[1]); f=sys.argv[2]; idx=int(sys.argv[3])
env = PR2PackedVariableCountEnv([k],[k])
obs, info = env.reset(seed=0, options={'object_count': k})
ap = GeneratedApproach(env.action_space, env.observation_space, None)
ap.reset(obs, info)
flat = np.load(f)[idx]
ap.t0 = time.time() - float(sys.argv[4]) if len(sys.argv)>4 else time.time()
pr = cProfile.Profile(); pr.enable()
r = ap.replan(flat)
pr.disable()
print('len', None if r is None else len(r[0]))
pstats.Stats(pr).sort_stats('cumulative').print_stats(30)
