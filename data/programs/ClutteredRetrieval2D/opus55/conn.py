import sys, numpy as np
from scipy import ndimage
from env_client import make_env
import approach; approach.DEBUG=False
from approach import *
seed=int(sys.argv[1]); steps=[int(s) for s in sys.argv[2].split(',')]
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
def conn(ap, r):
    names = [n for n in ap.obstacle_names(exclude=('target_block',))]
    o = Obstacles([ap.corners(n) for n in names])
    xs, clr = clearance_grid(o)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    free = clr >= r
    lab, _ = ndimage.label(free)
    tc = ap.center('target_block'); rc = ap.center('target_region')
    tm = free & (np.hypot(X-tc[0], Y-tc[1]) < 0.32)
    rm = free & (np.hypot(X-rc[0], Y-rc[1]) < 0.3)
    return bool(set(lab[tm].ravel()) & set(lab[rm].ravel()))
step=0
for target in steps:
    while step < target:
        obs, *_ = env.step(ap.get_action(obs)); step += 1
    ap._parse(obs)
    print(step, [ (r, conn(ap, r)) for r in (0.105, 0.12, 0.14, 0.16, 0.18)])
