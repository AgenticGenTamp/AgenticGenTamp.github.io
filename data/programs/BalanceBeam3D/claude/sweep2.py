import numpy as np, sys, json
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
R.drive(ss[0]-0.45, ss[1], 0.0)
R.move_to([ss[0],ss[1],0.30],fyaw=0.0,speed=0.03)
print("approach",ctrl.ee_world(R.obs),"ss",R.obs[38:45])
for z in [0.25,0.20,0.16,0.13,0.11,0.09,0.08,0.07,0.06,0.055,0.05]:
    R.move_to([ss[0],ss[1],z],fyaw=0.0,speed=0.015,tol=0.004,maxsteps=60)
    print(f" z={z:.3f} ee={ctrl.ee_world(R.obs)} blk={R.obs[bi:bi+3]} ssz={R.obs[40]:.4f} ssq={R.obs[41:45]} rew={R.rews[-1]}",flush=True)
json.dump(R.obs.tolist(),open("st_place.json","w"))
R.set_grip(0.0,n=20)
print("released blk",R.obs[bi:bi+3],"ss",R.obs[38:45],"rew",R.rews[-1])
R.move_to([ss[0],ss[1],0.25],speed=0.02)
for i in range(40):
    a=np.zeros(11); a[3:10]=R.arm_action(); a[10]=0.0; R.step(a)
print("final blk",R.obs[bi:bi+3],"ss",R.obs[38:45],"rew",R.rews[-1],"steps",R.steps)
json.dump(R.obs.tolist(),open("st_after.json","w"))
env.close()
