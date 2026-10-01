import numpy as np, sys, json
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1])
offs=[float(x) for x in sys.argv[2].split(",")]
order=[0,54,70]
REACH=0.45; CARRY_Z=0.26; SURF=0.047
env=make_env(); R=Runner(env,seed)
def yaw_of(q): return 2*np.arctan2(q[3],q[0])
ss=R.obs[38:41].copy()
for bi,dy in zip(order,offs):
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=yaw_of(R.obs[bi+3:bi+7])
    base=[blk[0]-REACH, blk[1], 0.0]
    R.go([blk[0],blk[1],0.14], base=base, fyaw=yaw, speed=0.05, tol=0.006, maxsteps=120)
    R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=60)
    R.set_grip(1.0)
    R.go([blk[0],blk[1],CARRY_Z], fyaw=yaw, speed=0.03, maxsteps=60, tol=0.01, settle=1)
    ty=ss[1]+dy
    zee = SURF+2*half-0.032+0.005
    R.go([REACH,0.0,CARRY_Z], base=[ss[0]-REACH,ty,0.0], rel=True, fyaw=0.0,
         speed=0.05, tol=0.02, maxsteps=80, settle=1, bvmax=0.04)
    print(f"  bi{bi} pre-descend step={R.steps} term={R.term_step}",flush=True)
    R.go([ss[0],ty,zee], fyaw=0.0, speed=0.02, tol=0.004, maxsteps=80)
    print(f"  bi{bi} pre-release step={R.steps} term={R.term_step}",flush=True)
    R.set_grip(0.0,n=10)
    print(f"  bi{bi} post-release step={R.steps} term={R.term_step} blk={R.obs[bi:bi+3]}",flush=True)
    R.go([ss[0],ty,CARRY_Z], speed=0.04, tol=0.02, maxsteps=50, settle=1)
    print(f"bi{bi} dy={dy}: blk={R.obs[bi:bi+3]} steps={R.steps} term={R.term_step}",flush=True)
for i in range(30):
    a=np.zeros(11); a[3:10]=R.arm_action(); a[10]=0.0; R.step(a)
print("blocks",R.obs[0:3],R.obs[54:57],R.obs[70:73])
print("TERM at",R.term_step,"total",R.steps)
env.close()
