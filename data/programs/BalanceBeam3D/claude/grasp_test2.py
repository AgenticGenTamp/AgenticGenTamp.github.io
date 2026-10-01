import numpy as np, sys, os
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1]); bi=int(sys.argv[2]) if len(sys.argv)>2 else 54
ZG=float(sys.argv[3]) if len(sys.argv)>3 else 0.005
env=make_env(); R=Runner(env,seed)
blk=R.obs[bi:bi+3].copy()
R.drive(blk[0]-0.45, blk[1], 0.0)
print("pre", R.move_to([R.obs[16]+0.35, R.obs[17], 0.30]))
print("above", R.move_to([blk[0],blk[1],0.15]), R.obs[bi:bi+3])
print("down", R.move_to([blk[0],blk[1],ZG],speed=0.006,tol=0.003), R.obs[bi:bi+3])
R.set_grip(1.0)
print("lift", R.move_to([blk[0],blk[1],0.25],speed=0.008))
b=R.obs[bi:bi+3]
print("BLK",b, "GRASPED" if b[2]>0.15 else "FAILED", "steps",R.steps)
env.close()
