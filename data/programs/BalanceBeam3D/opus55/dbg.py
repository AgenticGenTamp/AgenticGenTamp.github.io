import numpy as np
from run import run
np.set_printoptions(precision=3, suppress=True, linewidth=200)
te, t, tot, obs = run(0)
print(te, t)
for name, i in [("large",0),("seesaw",38),("sb1",54),("sb2",70)]:
    print(name, obs[i:i+7])
print("robot", obs[16:27])
