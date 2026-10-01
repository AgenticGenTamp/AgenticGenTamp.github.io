import numpy as np, sys
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1]); bi=int(sys.argv[2])
env=make_env(); R=Runner(env,seed)
def yaw_of(q): return 2*np.arctan2(q[3],q[0])
blk=R.obs[bi:bi+3].copy(); ss=R.obs[38:41].copy(); half=R.obs[bi+13]/2
yaw=yaw_of(R.obs[bi+3:bi+7])
R.drive(blk[0]-0.45, blk[1], 0.0)
R.move_to([blk[0],blk[1],0.16],fyaw=yaw)
R.move_to([blk[0],blk[1],0.005],fyaw=yaw,speed=0.012)
R.set_grip(1.0)
R.move_to([blk[0],blk[1],0.30],fyaw=yaw,speed=0.02)
print("grasped",R.obs[bi:bi+3],"steps",R.steps)
zt=0.0261+half+0.008
R.drive(ss[0]-0.45, ss[1]-0.20, 0.0)
R.move_to([ss[0],ss[1]-0.20,0.30],fyaw=0.0,speed=0.03)
R.move_to([ss[0],ss[1]-0.20,zt+0.10],fyaw=0.0,speed=0.02)
print("at start",ctrl.ee_world(R.obs),"rew",R.rews[-1],"steps",R.steps)
for k in range(21):
    y=ss[1]-0.20+0.02*k
    R.move_to([ss[0],y,zt+0.10],fyaw=0.0,speed=0.03,tol=0.006,maxsteps=40,base=[ss[0]-0.45,y,0.0])
    print(f"  y={y:.3f} blk={R.obs[bi:bi+3]} rew={R.rews[-1]:.4f}",flush=True)
env.close()
