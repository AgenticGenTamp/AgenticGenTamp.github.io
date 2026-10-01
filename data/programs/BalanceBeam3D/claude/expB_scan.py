import numpy as np, sys
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1]); axis=sys.argv[2]      # ypos yneg xpos xneg
REACH=0.45; CARRY_Z=0.26; SURF=0.047
env=make_env()
class R2(Runner):
    def __init__(s,*a,**k):
        super().__init__(*a,**k); s.snap=None
    def step(s,a):
        r,te=super().step(a)
        if te and s.snap is None: s.snap=(s.steps,s.obs.copy())
        return r,te
R=R2(env,seed)
ss=R.obs[38:41].copy()
print("SS",ss,"dims",R.obs[51:54],flush=True)
def place(bi,dy,dx=0.0):
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=2*np.arctan2(R.obs[bi+6],R.obs[bi+3])
    R.go([blk[0],blk[1],0.14], base=[blk[0]-REACH,blk[1],0.0], fyaw=yaw, speed=0.05, tol=0.006, maxsteps=120)
    R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=60)
    R.set_grip(1.0)
    R.go([blk[0],blk[1],CARRY_Z], fyaw=yaw, speed=0.03, maxsteps=60, tol=0.01, settle=1)
    ty=ss[1]+dy; tx=ss[0]+dx
    R.go([REACH,0.0,CARRY_Z], base=[tx-REACH,ty,0.0], rel=True, fyaw=0.0, speed=0.05, tol=0.02, maxsteps=80, settle=1, bvmax=0.04)
    R.go([tx,ty,SURF+2*half-0.032+0.005], fyaw=0.0, speed=0.02, tol=0.004, maxsteps=80)
    R.set_grip(0.0,n=10)
    R.go([tx,ty,CARRY_Z], speed=0.04, tol=0.02, maxsteps=50, settle=1)
place(0,0.0); print("b0 done",R.steps,R.obs[0:3],flush=True)
place(54,-0.10); print("b54 done",R.steps,R.obs[54:57],flush=True)
# pick block 70, carry high, sweep inward
bi=70; blk=R.obs[bi:bi+3].copy(); yaw=2*np.arctan2(R.obs[bi+6],R.obs[bi+3])
R.go([blk[0],blk[1],0.14], base=[blk[0]-REACH,blk[1],0.0], fyaw=yaw, speed=0.05, tol=0.006, maxsteps=120)
R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=60)
R.set_grip(1.0)
R.go([blk[0],blk[1],CARRY_Z], fyaw=yaw, speed=0.03, maxsteps=60, tol=0.01, settle=1)
print("picked b70",R.steps,flush=True)
if axis in ("ypos","yneg"):
    sgn=1.0 if axis=="ypos" else -1.0
    seq=[round(0.30-0.01*i,3) for i in range(31)]
    for d in seq:
        ty=ss[1]+sgn*d; tx=ss[0]
        R.go([REACH,0.0,CARRY_Z], base=[tx-REACH,ty,0.0], rel=True, fyaw=0.0, speed=0.05, tol=0.015, maxsteps=120, settle=3, bvmax=0.04)
        print(f"hover d={sgn*d:+.3f} step={R.steps} blk={R.obs[70:73]} term={R.snap[0] if R.snap else None}",flush=True)
        if R.snap: break
else:
    sgn=1.0 if axis=="xpos" else -1.0
    for d in [round(0.20-0.01*i,3) for i in range(21)]:
        tx=ss[0]+sgn*d; ty=ss[1]
        R.go([REACH,0.0,CARRY_Z], base=[tx-REACH,ty,0.0], rel=True, fyaw=0.0, speed=0.05, tol=0.015, maxsteps=120, settle=3, bvmax=0.04)
        print(f"hover dx={sgn*d:+.3f} step={R.steps} blk={R.obs[70:73]} term={R.snap[0] if R.snap else None}",flush=True)
        if R.snap: break
if R.snap: print("TERM",R.snap[0],"blocks",R.snap[1][0:3],R.snap[1][54:57],R.snap[1][70:73])
else: print("NO TERM")
env.close()
