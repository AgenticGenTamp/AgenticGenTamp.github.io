import numpy as np, sys
import ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
REACH=0.45; CARRY_Z=0.26; SURF=0.047
BI=54   # small block 1
def yaw_of(q): return 2*np.arctan2(q[3],q[0])
def sspose(o):
    p=o[38:41]; q=o[41:45]
    yaw=2*np.arctan2(q[3],q[0])
    roll=2*np.arctan2(q[1],q[0]); pitch=2*np.arctan2(q[2],q[0])
    return np.array([p[0],p[1],p[2],np.degrees(yaw),np.degrees(roll),np.degrees(pitch)])
def run(seed, dz, xoff, bi=BI, label=""):
    env=make_env(); R=Runner(env,seed)
    ss0=sspose(R.obs); sxy=R.obs[38:41].copy()
    blk=R.obs[bi:bi+3].copy(); half=R.obs[bi+13]/2
    yaw=yaw_of(R.obs[bi+3:bi+7])
    base=[blk[0]-REACH, blk[1], 0.0]
    R.go([blk[0],blk[1],0.14], base=base, fyaw=yaw, speed=0.05, tol=0.006, maxsteps=140)
    R.go([blk[0],blk[1],0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=70)
    R.set_grip(1.0,n=18)
    R.go([blk[0],blk[1],CARRY_Z], fyaw=yaw, speed=0.03, maxsteps=70, tol=0.008)
    ee=ctrl.ee_world(R.obs); gdz=R.obs[bi+2]-ee[2]
    tx=sxy[0]+xoff; ty=sxy[1]
    R.go([REACH,0.0,CARRY_Z], base=[sxy[0]+xoff-REACH,ty,0.0], rel=True, fyaw=0.0,
         speed=0.05, tol=0.02, maxsteps=110, settle=1, bvmax=0.04)
    R.go([tx,ty,0.12], fyaw=0.0, speed=0.03, tol=0.005, maxsteps=90)
    ss_carry=sspose(R.obs)
    tz=max(SURF+0.010, SURF+half-gdz+dz)
    R.go([tx,ty,tz], fyaw=0.0, speed=0.015, tol=0.003, maxsteps=110)
    ee_d=ctrl.ee_world(R.obs)
    ss_desc=sspose(R.obs); cube_pre=R.obs[bi:bi+3].copy()
    R.set_grip(0.0,n=16)
    ss_open=sspose(R.obs)
    R.go([tx,ty,0.20], speed=0.03, tol=0.02, maxsteps=70, settle=1)
    for i in range(60):
        a=np.zeros(11); a[3:10]=R.arm_action(); a[10]=0.0; R.step(a)
    ss_f=sspose(R.obs); cube=R.obs[bi:bi+3].copy()
    print(f"== {label} dz={dz:+.3f} (tool z={tz:.3f}) xoff={xoff:+.3f} seed={seed} GRIP_DZ={gdz:.4f}")
    print(f"   ee_at_descent={ee_d}  cube_before_release={cube_pre}")
    print(f"   ss0     {ss0}")
    print(f"   d_carry {ss_carry-ss0}")
    print(f"   d_desc  {ss_desc-ss0}")
    print(f"   d_open  {ss_open-ss0}")
    print(f"   d_final {ss_f-ss0}")
    print(f"   cube_final={cube}  slide_xy=({cube[0]-tx:+.4f},{cube[1]-ty:+.4f}) z={cube[2]:.4f} on_plank={abs(cube[2]-(SURF+half))<0.006}")
    print(f"   steps={R.steps}",flush=True)
    env.close()
if __name__=="__main__":
    lab=sys.argv[1]; dz=float(sys.argv[2]); xoff=float(sys.argv[3])
    seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
    bi=int(sys.argv[5]) if len(sys.argv)>5 else BI
    run(seed,dz,xoff,bi,lab)
