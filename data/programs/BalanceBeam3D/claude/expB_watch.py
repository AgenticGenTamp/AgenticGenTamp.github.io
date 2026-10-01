import numpy as np, sys
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1]); offs=[float(x) for x in sys.argv[2].split(",")]
order=[0,54,70]
REACH=0.45; CARRY_Z=0.26; SURF=0.047
env=make_env()
class R2(Runner):
    def __init__(s,*a,**k):
        super().__init__(*a,**k); s.hist=[]; s.snap=None
    def step(s,a):
        prev=s.obs.copy()
        r,te=super().step(a)
        s.hist.append(np.concatenate([[s.steps],s.obs[0:3],s.obs[54:57],s.obs[70:73],s.obs[38:45]]))
        if te and s.snap is None: s.snap=(s.steps,prev.copy(),s.obs.copy())
        return r,te
R=R2(env,seed)
ss=R.obs[38:41].copy()
print("SS pos",ss,"quat",R.obs[41:45],"dims?",R.obs[51:54] if len(R.obs)>53 else None)
for bi,dy in zip(order,offs):
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=2*np.arctan2(R.obs[bi+6],R.obs[bi+3])
    base=[blk[0]-REACH, blk[1], 0.0]
    R.go([blk[0],blk[1],0.14], base=base, fyaw=yaw, speed=0.05, tol=0.006, maxsteps=120)
    R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=60)
    R.set_grip(1.0)
    R.go([blk[0],blk[1],CARRY_Z], fyaw=yaw, speed=0.03, maxsteps=60, tol=0.01, settle=1)
    ty=ss[1]+dy
    zee = SURF+2*half-0.032+0.005
    R.go([REACH,0.0,CARRY_Z], base=[ss[0]-REACH,ty,0.0], rel=True, fyaw=0.0,
         speed=0.05, tol=0.02, maxsteps=80, settle=1, bvmax=0.04)
    R.go([ss[0],ty,zee], fyaw=0.0, speed=0.02, tol=0.004, maxsteps=80)
    R.set_grip(0.0,n=10)
    R.go([ss[0],ty,CARRY_Z], speed=0.04, tol=0.02, maxsteps=50, settle=1)
    if R.snap: break
if R.snap:
    t,prev,now=R.snap
    print("TERM step",t)
    print(" prev blocks",prev[0:3],prev[54:57],prev[70:73],"ss",prev[38:45])
    print(" now  blocks",now[0:3],now[54:57],now[70:73],"ss",now[38:45])
    H=np.array(R.hist)
    for row in H[max(0,len(H)-12):]:
        print("  h",row)
else:
    print("NO TERM")
env.close()
