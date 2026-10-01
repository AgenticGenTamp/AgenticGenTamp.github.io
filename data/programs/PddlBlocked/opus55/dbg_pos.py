import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
sd = int(sys.argv[1]); T = int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=sd)
ap = GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(obs, info)
def show(tag):
    for n in obs.get_object_names():
        if n == 'blocker' or n.startswith('green'):
            o = obs.get_object_from_name(n); print(tag, n, [round(float(obs.get(o, f)), 3) for f in ('pose_x', 'pose_y', 'pose_z')])
show('init')
for t in range(T):
    obs, *_ = env.step(ap.get_action(obs))
show('t%d' % T)
