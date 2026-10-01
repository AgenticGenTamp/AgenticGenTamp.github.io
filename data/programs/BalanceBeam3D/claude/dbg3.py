import numpy as np
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
env=make_env(); R=Runner(env,0)
blk=R.obs[54:57].copy()
R.drive(blk[0]-0.45, blk[1], 0.0)
for g in [[R.obs[16]+0.35,R.obs[17],0.30],[blk[0],blk[1],0.15]]:
    ee,i=R.move_to(g)
    print("goal",np.round(g,4),"ee",ee,"ee_des",R.ee_des(),"i",i,
          "qerr",np.round(R.q_des-R.obs[19:26],4),"qi",np.round(R.qi,3))
env.close()
