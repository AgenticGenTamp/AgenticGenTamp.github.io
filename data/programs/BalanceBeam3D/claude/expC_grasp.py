import numpy as np, sys, time
import kinova, ctrl
from env_client import make_env
from runner import Runner

def yaw_of(q): return 2*np.arctan2(q[3],q[0])

def wrist_R(obs):
    return ctrl.Rz(obs[18]) @ kinova.fk(obs[19:26], ctrl.TOOL)[:3,:3]

def trial(seed, bi, LAT, PERP=0.0):
    env=make_env(); R=Runner(env,seed)
    blk=R.obs[bi:bi+3].copy()
    yaw=yaw_of(R.obs[bi+3:bi+7])
    base=[blk[0]-0.45, blk[1], 0.0]
    R.go([blk[0],blk[1],0.14], base=base, fyaw=yaw, speed=0.05, tol=0.006, maxsteps=120)
    Rw=wrist_R(R.obs); fy=np.arctan2(Rw[1,0],Rw[0,0])
    ax=blk[0]+LAT*np.cos(fy)-PERP*np.sin(fy)
    ay=blk[1]+LAT*np.sin(fy)+PERP*np.cos(fy)
    R.go([ax,ay,0.14], base=base, fyaw=yaw, speed=0.05, tol=0.004, maxsteps=50)
    R.go([ax,ay,0.005], fyaw=yaw, speed=0.02, tol=0.003, maxsteps=60)
    knock=float(np.linalg.norm(R.obs[bi:bi+2]-blk[:2]))
    R.set_grip(1.0)
    R.go([ax,ay,0.26], fyaw=yaw, speed=0.02, tol=0.004, maxsteps=80)
    ok=bool(R.obs[bi+2]>0.15)
    Rw2=wrist_R(R.obs)
    off=Rw2.T@(R.obs[bi:bi+3]-ctrl.ee_world(R.obs))
    env.close()
    return ok, knock, off, fy, R.steps

if __name__=="__main__":
    LAT=float(sys.argv[1]); s0=int(sys.argv[2]); s1=int(sys.argv[3])
    PERP=float(sys.argv[4]) if len(sys.argv)>4 else 0.0
    t0=time.time()
    for seed in range(s0,s1):
        for bi in (0,54,70):
            try:
                ok,kn,off,fy,st=trial(seed,bi,LAT,PERP)
                print(f"LAT={LAT} PERP={PERP} seed{seed} bi{bi} ok={int(ok)} knock={kn:.4f} off=[{off[0]:.4f},{off[1]:.4f},{off[2]:.4f}] fy={fy:.3f} steps={st}",flush=True)
            except Exception as e:
                print(f"LAT={LAT} PERP={PERP} seed{seed} bi{bi} ERR {type(e).__name__} {e}",flush=True)
    print(f"# done {time.time()-t0:.0f}s",flush=True)
