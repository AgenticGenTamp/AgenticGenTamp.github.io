import numpy as np, sys, os
import kinova, ctrl
from env_client import make_env
from runner import Runner
def yaw_of(q):
    return 2*np.arctan2(q[3],q[0])
def trial(seed,bi,fy=True):
    env=make_env(); R=Runner(env,seed)
    blk=R.obs[bi:bi+3].copy()
    yaw=yaw_of(R.obs[bi+3:bi+7]) if fy else None
    R.drive(blk[0]-0.45, blk[1], 0.0)
    R.move_to([blk[0],blk[1],0.16],fyaw=yaw)
    R.move_to([blk[0],blk[1],0.005],fyaw=yaw,speed=0.012)
    moved=np.linalg.norm(R.obs[bi:bi+2]-blk[:2])
    R.set_grip(1.0)
    R.move_to([blk[0],blk[1],0.25],fyaw=yaw,speed=0.015)
    ok=R.obs[bi+2]>0.15
    env.close()
    return ok, moved, R.steps
if __name__=="__main__":
    tot=0; n=0
    for seed in range(int(sys.argv[1]),int(sys.argv[2])):
        for bi in (0,54,70):
            ok,moved,st=trial(seed,bi)
            tot+=ok; n+=1
            print(f"seed{seed} bi{bi} {'OK ' if ok else 'FAIL'} knock={moved:.3f} steps={st}",flush=True)
    print("SUCCESS",tot,"/",n)
