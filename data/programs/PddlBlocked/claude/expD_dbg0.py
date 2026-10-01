import numpy as np
from expD_lib import *; from expD_blk import remove_blocker2
env = make_env()
ap, obs, ns, ok, why = remove_blocker2(env, 0)
r = ap._robot(obs); B = ap._blocks(obs)
print("ok",ok,"why",why,"steps",ns,"holding",r["holding"],"base",np.round(r["base"],3))
print("blocker",np.round(B["blocker"],3),"g0",np.round(B["green0"],3))
print("q",np.round(r["q"],3))
qh = home_q(env,0); print("qhome",np.round(qh,3))
