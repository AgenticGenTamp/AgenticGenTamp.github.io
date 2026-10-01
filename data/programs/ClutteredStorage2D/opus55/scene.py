import sys
from env_client import make_env
from approach import GeneratedApproach
seed = int(sys.argv[1])
env = make_env()
obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap._parse(obs)
print('shelf x1', ap.sx1, 'w1', ap.sw1, 'robot', ap.rx, ap.ry)
for n,b in ap.blocks.items(): print(n, 'th', round(b['th'],3), b['poly'].round(3).tolist())
print(ap._grasp_options('block1')[:3])
