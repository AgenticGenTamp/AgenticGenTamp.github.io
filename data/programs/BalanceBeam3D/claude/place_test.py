import numpy as np, sys
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1]); bi=int(sys.argv[2]); dy=float(sys.argv[3])
env=make_env(); R=Runner(env,seed)
blk=R.obs[bi:bi+3].copy(); ss=R.obs[38:41].copy()
half=R.obs[bi+13]/2
print("blk",blk,"ss",ss)
R.drive(blk[0]-0.45, blk[1], 0.0)
print("above",R.move_to([blk[0],blk[1],0.16]),R.obs[bi:bi+3])
print("down",R.move_to([blk[0],blk[1],0.005],speed=0.012))
R.set_grip(1.0)
print("lift",R.move_to([blk[0],blk[1],0.25],speed=0.015))
tx,ty = ss[0], ss[1]+dy
R.drive(tx-0.45, ty, 0.0)
print("carry",R.move_to([tx,ty,0.25]))
zt = 0.0261+half+0.012
print("lower",R.move_to([tx,ty,zt],speed=0.01))
R.set_grip(0.0,n=15)
print("blk after release",R.obs[bi:bi+3],"rew",R.rews[-1])
print("retreat",R.move_to([tx,ty,0.25],speed=0.015))
for i in range(30):
    a=np.zeros(11); a[3:10]=R.arm_action(); a[10]=0.0; R.step(a)
print("blk",R.obs[bi:bi+3],"ss q",R.obs[41:45],"rew",R.rews[-1],"steps",R.steps)
env.close()
