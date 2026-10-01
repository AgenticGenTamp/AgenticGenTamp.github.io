import numpy as np, math, time
from expD_lib import *; from expD_core import *; from expD_blk import remove_blocker2
env = make_env()
for seed in [0,3]:
    ap, obs, ns, ok, why = remove_blocker2(env, seed)
    b = ap._blocks(obs)
    print(f"seed={seed} remove_blocker2 ok={ok} why={why} steps={ns} blocker={np.round(b['blocker'],3)} g0={np.round(b['green0'],3)}")
