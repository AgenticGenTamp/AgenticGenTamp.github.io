import numpy as np
from env_client import make_env
from ctrl import Bot
np.set_printoptions(precision=4,suppress=True)
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy()
Z=0.593
b.grip(1.0,5)
b.servo(np.array([can[0]+0.05,can[1]+0.17,0.70]),1.0,200,tol=0.004)
b.servo(np.array([can[0]+0.05,can[1]+0.17,Z]),1.0,140,tol=0.004)
e,err=b.servo(np.array([can[0]+0.05,-0.345,Z]),1.0,60,tol=0.002)
print("pre-close EE",e,"can",b.obs[SL:SL+3])
# CLOSE, log per step
p=b.obs[SL:SL+3].copy()
a=np.zeros(11); a[10]=0.0
traj=[]
for i in range(30):
    b.step(a); traj.append(np.linalg.norm(b.obs[SL:SL+3]-p))
print("can displacement per step after CLOSE cmd:", np.round(traj[:16],5))
print("obs103 now",b.obs[103])
z2=b.obs[SL+2]
e2,_=b.servo(np.array([e[0],e[1],Z+0.22]),0.0,110,tol=0.005)
print("after lift: can=%s dz=%+.4f EE=%s"%(b.obs[SL:SL+3],b.obs[SL+2]-z2,e2))
if b.obs[SL+2]-z2>0.03:
    bx=b.obs[93]; p2=b.obs[SL:SL+3].copy()
    a=np.zeros(11); a[0]=-0.1; a[10]=0.0
    for _ in range(20): b.step(a)
    print("*** BASE-DRIVE CONFIRM: base dx=%.4f can d=%s"%(b.obs[93]-bx,b.obs[SL:SL+3]-p2))
    # release
    a=np.zeros(11); a[10]=1.0; zr=b.obs[SL+2]
    for _ in range(25): b.step(a)
    print("*** RELEASE (1.0): can dz=%+.4f can=%s"%(b.obs[SL+2]-zr,b.obs[SL:SL+3]))
print("steps",b.n)
b.env.close()
