import numpy as np, sys
import ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
REACH=0.45
def yaw_of(q): return 2*np.arctan2(q[3],q[0])
def measure(seed, bi, name):
    env=make_env(); R=Runner(env,seed)
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=yaw_of(R.obs[bi+3:bi+7])
    base=[blk[0]-REACH, blk[1], 0.0]
    R.go([blk[0],blk[1],0.14], base=base, fyaw=yaw, speed=0.05, tol=0.006, maxsteps=140)
    R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=70)
    ee_pre=ctrl.ee_world(R.obs)
    R.set_grip(1.0,n=20)
    ee_c=ctrl.ee_world(R.obs); c_c=R.obs[bi:bi+3].copy()
    R.go([blk[0],blk[1],0.26], fyaw=yaw, speed=0.03, maxsteps=80, tol=0.006)
    ee=ctrl.ee_world(R.obs); c=R.obs[bi:bi+3].copy()
    print(f"{name} bi={bi} size={R.obs[bi+13]:.4f}")
    print(f"  pre-close ee={ee_pre}")
    print(f"  closed(no lift) ee_z={ee_c[2]:.4f} cube_z={c_c[2]:.4f} dz={c_c[2]-ee_c[2]:.4f}")
    print(f"  lifted ee={ee} cube={c}")
    print(f"  GRIP_DZ={c[2]-ee[2]:.4f}  dxy=({c[0]-ee[0]:+.4f},{c[1]-ee[1]:+.4f})")
    # second lift height to confirm
    R.go([blk[0],blk[1],0.18], fyaw=yaw, speed=0.03, maxsteps=60, tol=0.006)
    ee2=ctrl.ee_world(R.obs); c2=R.obs[bi:bi+3].copy()
    print(f"  @z0.18 GRIP_DZ={c2[2]-ee2[2]:.4f} steps={R.steps}",flush=True)
    env.close()
for s in [0,1]:
    measure(s,54,f"SMALL(seed{s})")
    measure(s,0,f"LARGE(seed{s})")
