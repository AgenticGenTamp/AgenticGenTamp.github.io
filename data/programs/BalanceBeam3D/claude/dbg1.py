import numpy as np, sys
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
env=make_env(); R=Runner(env,0)
blk=R.obs[54:57].copy(); print("blk",blk,"base",R.obs[16:19])
R.drive(blk[0]-0.45, blk[1], 0.0); print("steps",R.steps,"base",R.obs[16:19])
print("ee now", ctrl.ee_world(R.obs))
for g in [[R.obs[16]+0.35, R.obs[17], 0.30],[blk[0],blk[1],0.15],[blk[0],blk[1],0.005]]:
    ee,i=R.move_to(g, speed=0.015)
    print("goal",np.round(g,4),"ee",ee,"iters",i,"tot",R.steps,"blk",R.obs[54:57])
env.close()
