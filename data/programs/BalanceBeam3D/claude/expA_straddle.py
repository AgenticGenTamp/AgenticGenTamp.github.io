import numpy as np, sys
import ctrl
from env_client import make_env
from runner import Runner
np.set_printoptions(precision=4, suppress=True)
REACH=0.45; SURF=0.047
def sspose(o):
    p=o[38:41]; q=o[41:45]
    return np.array([p[0],p[1],p[2],np.degrees(2*np.arctan2(q[3],q[0])),
                     np.degrees(2*np.arctan2(q[1],q[0])),np.degrees(2*np.arctan2(q[2],q[0]))])
tz=float(sys.argv[1]); xoff=float(sys.argv[2]); seed=int(sys.argv[3])
env=make_env(); R=Runner(env,seed); R.set_grip(0.0,n=6)
ss0=sspose(R.obs); s=R.obs[38:41].copy()
tx=s[0]+xoff; ty=s[1]
R.go([REACH,0.0,0.20], base=[tx-REACH,ty,0.0], rel=True, fyaw=0.0, speed=0.05, tol=0.02, maxsteps=140, settle=1, bvmax=0.05)
R.go([tx,ty,0.12], fyaw=0.0, speed=0.03, tol=0.005, maxsteps=90)
print("approach d_ss",sspose(R.obs)-ss0)
R.go([tx,ty,tz], fyaw=0.0, speed=0.012, tol=0.003, maxsteps=140)
print(f"EMPTY-OPEN straddle tool_z={tz:.3f} (SURF{tz-SURF:+.3f}) xoff={xoff:+.3f} seed={seed}")
print("  ee=",ctrl.ee_world(R.obs))
print("  d_desc ",sspose(R.obs)-ss0)
R.go([tx,ty,0.20], speed=0.03, tol=0.02, maxsteps=80, settle=1)
for i in range(40):
    a=np.zeros(11); a[3:10]=R.arm_action(); a[10]=0.0; R.step(a)
print("  d_final",sspose(R.obs)-ss0, "steps",R.steps,flush=True)
env.close()
