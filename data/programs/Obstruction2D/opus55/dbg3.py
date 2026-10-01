import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import approach as A
seed, count = int(sys.argv[1]), int(sys.argv[2])
env = make_env()
obs, info = env.reset(seed=seed, options=({'object_count': count} if count>0 else None))
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info); ap._parse(obs)
s=ap.surface; b=ap.block
lo = s["x"] + 0.003; hi = s["x"] + s["w"] - b["w"] - 0.003
for dx in np.linspace(lo, hi, 7):
    glo = max(0.006, A.X_MIN - dx); ghi = min(b["w"] - 0.006, A.X_MAX - dx)
    grel = min(max(b["w"] / 2, glo), ghi)
    print(round(dx,4), round(grel,4), [o['name'] for o in ap.obst.values() if ap._column_hit(o, dest_x=dx, grel=grel)])
