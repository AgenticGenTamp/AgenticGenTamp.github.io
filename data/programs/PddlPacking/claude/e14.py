import numpy as np, fk, ctrl, lib
from env_client import make_env
import sys
ang_off = float(sys.argv[1]) if len(sys.argv)>1 else 0.0
env=make_env(); R=lib.Runner(env,seed=0)
bl=ctrl.blocks(R.obs); tb=bl["block1"]
R.move_base((tb[0]-0.55,tb[1]-0.2,0.0))
bx,by,_=R.rob()[:3]
ang=ctrl.align_angle(ctrl.quat_yaw(tb[3:7]))+ang_off
z=tb[2]-0.05
ok=R.cart(np.array([tb[0]-bx,tb[1]-by,z]),ang,maxit=40)
print("init",ok,np.round(R.tool(),4))
while z>tb[2]-0.40:
    z-=0.01
    ok=R.cart(np.array([tb[0]-bx,tb[1]-by,z]),ang,maxit=4,step_len=0.012,tol=1.5e-3)
    p=R.tool()
    rej,t=R.step([0]*10+[-1.0]); ga=R.rob()[11]
    print(round(z,3),"ok",ok,"tool",np.round(p,4),"grasp",ga)
    if ga:
        print("GTF",np.round(ctrl.gtf(R.obs),5)); break
    R.step([0]*10+[1.0])
    if not ok and abs(p[2]-z)>0.05: break
env.close()
