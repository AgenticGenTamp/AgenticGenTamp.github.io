import numpy as np, math
from expD_lib import *
env = make_env()
for seed in range(4):
    ap, obs, n = remove_blocker(env, seed)
    b = ap._blocks(obs); r = ap._robot(obs)
    g0 = b["green0"]
    print(f"seed={seed} n={n} dir={np.round(ap.dir,3)} yaw={math.atan2(ap.dir[1],ap.dir[0]):+.3f} "
          f"green0={np.round(g0,3)} base_now={np.round(r['base'],3)} holding={r['holding']}")
    print("   surfaces:", {k: np.round(v,3).tolist() for k,v in ap._surfaces(obs).items()})
    print("   blocks:", {k: np.round(v,3).tolist() for k,v in b.items()})
