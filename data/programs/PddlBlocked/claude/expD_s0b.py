import numpy as np, math
from expD_lib import *
env = make_env()
for seed in range(4):
    ap, obs, n = remove_blocker(env, seed, max_steps=200)
    b = ap._blocks(obs); r = ap._robot(obs)
    print(f"seed={seed} n={n} task_i={ap.task_i} dir={np.round(ap.dir,3)} yaw={math.atan2(ap.dir[1],ap.dir[0]):+.3f} "
          f"green0={np.round(b['green0'],3)} blocker={np.round(b['blocker'],3)} base={np.round(r['base'],3)}")
