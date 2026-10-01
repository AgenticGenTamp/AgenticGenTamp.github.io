import numpy as np, fk, ctrl, lib
from env_client import make_env
env=make_env(); R=lib.Runner(env,seed=0)
bl=ctrl.blocks(R.obs); tb=bl["block1"]
print("base",R.move_base((tb[0]-0.55,tb[1]-0.2,0.0)))
bx,by,_=R.rob()[:3]
ang=ctrl.align_angle(ctrl.quat_yaw(tb[3:7]))
res=[]
for dz in np.arange(-0.20,-0.56,-0.02):
    tp=np.array([tb[0]-bx,tb[1]-by,tb[2]+dz])
    ok=R.cart(tp,ang,maxit=30)
    rej,t=R.step([0]*10+[-1.0])
    ga=R.rob()[11]
    res.append((round(dz,3),ok,ga))
    print(round(dz,3),"cartok",ok,"grasp",ga, np.round(ctrl.gtf(R.obs),4) if ga else "")
    if ga: break
    R.step([0]*10+[1.0])
print("steps",R.n)
env.close()
