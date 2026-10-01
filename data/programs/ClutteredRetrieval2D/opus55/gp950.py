import sys
import approach
orig = approach.grid_path
def gp(m, a, b):
    r = orig(m, a, b); print('grid', m.held is not None, None if r is None else len(r), a[:2].round(2), b[:2].round(2)); return r
approach.grid_path = gp
from env_client import make_env
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]))
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(1000):
    obs, r, term, trunc, info = env.step(ap.get_action(obs))
    if term: print("SOLVED", step + 1); break
