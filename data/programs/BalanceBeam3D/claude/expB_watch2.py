import numpy as np, sys
import kinova, ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
seed=int(sys.argv[1]); offs=[float(x) for x in sys.argv[2].split(",")]
order=[0,54,70]; REACH=0.45; CARRY_Z=0.26; SURF=0.047
def tilt(q):  # deg away from pure-z rotation
    return float(np.degrees(2*np.arcsin(min(1.0,np.hypot(q[1],q[2])))))
env=make_env()
class R2(Runner):
    def __init__(s,*a,**k):
        super().__init__(*a,**k); s.rec=[]; s.snap=None
    def step(s,a):
        r,te=super().step(a)
        s.rec.append((s.steps,tilt(s.obs[41:45]),s.obs[2],s.obs[56],s.obs[72],s.obs[1],s.obs[55],s.obs[71]))
        if te and s.snap is None: s.snap=(s.steps,s.obs.copy())
        return r,te
R=R2(env,seed); ss=R.obs[38:41].copy()
for bi,dy in zip(order,offs):
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=2*np.arctan2(R.obs[bi+6],R.obs[bi+3])
    R.go([blk[0],blk[1],0.14], base=[blk[0]-REACH,blk[1],0.0], fyaw=yaw, speed=0.05, tol=0.006, maxsteps=120)
    R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=60)
    R.set_grip(1.0)
    R.go([blk[0],blk[1],CARRY_Z], fyaw=yaw, speed=0.03, maxsteps=60, tol=0.01, settle=1)
    ty=ss[1]+dy
    R.go([REACH,0.0,CARRY_Z], base=[ss[0]-REACH,ty,0.0], rel=True, fyaw=0.0, speed=0.05, tol=0.02, maxsteps=80, settle=1, bvmax=0.04)
    R.go([ss[0],ty,SURF+2*half-0.032+0.005], fyaw=0.0, speed=0.02, tol=0.004, maxsteps=80)
    R.set_grip(0.0,n=10)
    R.go([ss[0],ty,CARRY_Z], speed=0.04, tol=0.02, maxsteps=50, settle=1)
    print(f"placed bi{bi} dy={dy} step={R.steps} tilt={R.rec[-1][1]:.2f} term={R.snap[0] if R.snap else None}",flush=True)
for i in range(60):
    a=np.zeros(11); a[3:10]=R.arm_action(); R.step(a)
rec=np.array(R.rec)
print("final tilt",rec[-1][1],"final ys",rec[-1][5:8],"final zs",rec[-1][2:5])
if R.snap:
    t=R.snap[0]; k=t-1
    print(f"TERM step {t} tilt_at_term={rec[k][1]:.3f} tilt_prev={rec[k-1][1]:.3f} tilt_next={rec[min(k+1,len(rec)-1)][1]:.3f}")
    print(" blocks at term y",rec[k][5:8],"z",rec[k][2:5])
else:
    print("NO TERM; min tilt after last place:",rec[-200:,1].min())
env.close()
