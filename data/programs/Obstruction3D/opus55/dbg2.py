import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from kin import fk
seed=int(sys.argv[1]); N=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(N):
    a = ap.get_action(obs); obs, *_ = env.step(a)
ap._read(obs)
np.set_printoptions(precision=3, suppress=True)
print('q', ap.q, 'base', ap.base, 'last a', a)
for n,(p,he) in ap.objs.items(): print(n, p, he)
qt,e = ap._ik(np.array([0.337,-0.362,0.15]), ap.task['yaw']); print('ik target q', qt, e, 'yaw', ap.task['yaw'])
