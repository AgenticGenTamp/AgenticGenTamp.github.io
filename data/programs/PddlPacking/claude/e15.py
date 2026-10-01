import numpy as np, fk, ctrl, lib
from env_client import make_env
env=make_env(); R=lib.Runner(env,seed=0)
bl=ctrl.blocks(R.obs); tb=bl["block1"]
R.move_base((tb[0]-0.55,tb[1]-0.2,0.0))
bx,by,_=R.rob()[:3]
ang=ctrl.align_angle(ctrl.quat_yaw(tb[3:7]))
R.cart(np.array([tb[0]-bx,tb[1]-by,tb[2]-0.27]),ang,maxit=60)
rej,t=R.step([0]*10+[-1.0])
print("grasp",R.rob()[11], np.round(ctrl.gtf(R.obs),5))
if R.rob()[11]<0.5: raise SystemExit("no grasp")
data=[]
rng=np.random.default_rng(0)
def rec():
    r=R.rob(); b=ctrl.blocks(R.obs)["block1"]
    data.append((r[:10].copy(), b[:7].copy()))
rec()
for k in range(60):
    a=np.zeros(11); a[3:10]=rng.uniform(-0.2,0.2,7)
    rej,t=R.step(a)
    if not rej: rec()
print("n",len(data))
np.save("calib.npy", np.array([np.concatenate([d[0],d[1]]) for d in data]))
env.close()
