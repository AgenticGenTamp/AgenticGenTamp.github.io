import time, sys
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]))
ap = GeneratedApproach(env.action_space, env.observation_space, {})
t=time.time(); ap.reset(obs, info); print('plan', time.time()-t, 'margin', ap.margin, 'len', None if ap.path is None else len(ap.path))
t=time.time(); a=ap.get_action(obs); print('act', time.time()-t, a)
