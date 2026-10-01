import numpy as np, sys, json
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1])
offs=[float(x) for x in sys.argv[2].split(",")]   # y-offsets for [large, s1, s2]
order=[0,54,70]
env=make_env(); R=Runner(env,seed)
def yaw_of(q): return 2*np.arctan2(q[3],q[0])
ss=R.obs[38:41].copy()
SURF=0.047
term=False
for bi,dy in zip(order,offs):
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=yaw_of(R.obs[bi+3:bi+7])
    R.drive(blk[0]-0.45, blk[1], 0.0)
    R.move_to([blk[0],blk[1],0.16],fyaw=yaw)
    R.move_to([blk[0],blk[1],0.005],fyaw=yaw,speed=0.012)
    R.set_grip(1.0)
    R.move_to([blk[0],blk[1],0.30],fyaw=yaw,speed=0.02)
    ty=ss[1]+dy
    R.drive(ss[0]-0.45, ty, 0.0)
    R.move_to([ss[0],ty,0.30],fyaw=0.0,speed=0.03)
    # EE z such that block bottom just above surface: block center = ee+0.032
    zee = SURF+2*half-0.032+0.004
    R.move_to([ss[0],ty,zee],fyaw=0.0,speed=0.012,tol=0.004,maxsteps=120)
    R.set_grip(0.0,n=15)
    R.move_to([ss[0],ty,0.30],speed=0.02)
    print(f"placed bi{bi} at dy={dy}: blk={R.obs[bi:bi+3]} ss={R.obs[38:41]} ssq={R.obs[41:45]} rew={R.rews[-1]} steps={R.steps}",flush=True)
# settle
for i in range(40):
    a=np.zeros(11); a[3:10]=R.arm_action(); a[10]=0.0
    r,te=R.step(a)
    if te: term=True
print("FINAL rew",R.rews[-1],"term",term,"steps",R.steps)
print("blocks",R.obs[0:3],R.obs[54:57],R.obs[70:73])
json.dump(R.obs.tolist(),open(f"st_p3_{seed}.json","w"))
env.close()
