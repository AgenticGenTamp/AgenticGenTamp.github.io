import sys, numpy as np
from env_client import make_env
from ctrl import Bot
np.set_printoptions(precision=4,suppress=True)
G=float(sys.argv[1]); ZBLK=float(sys.argv[2])   # table-contact FK z for this G
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy()
b.grip(G,5)
tgt=np.array([can[0]-0.45, can[1]])
for _ in range(60):
    d=tgt-b.obs[93:95]
    if np.linalg.norm(d)<0.01: break
    a=np.zeros(11); a[0:2]=np.clip(d/0.87,-0.1,0.1); a[2]=np.clip(-b.obs[95],-0.1,0.1); a[10]=G
    b.step(a)
Z = ZBLK+0.055   # fingertips ~5.5cm above table = can mid height
y0 = can[1]+0.26
e,err=b.servo(np.array([can[0],y0,0.72]),G,200,tol=0.004)
e,err=b.servo(np.array([can[0],y0,Z]),G,120,tol=0.004)
print("G=%.1f start EE %s err %.4f Z=%.4f"%(G,e,err,Z))
p0=b.obs[SL:SL+3].copy(); hit=None
for k in range(1,55):
    y=y0-0.005*k
    e,err=b.servo(np.array([can[0],y,Z]),G,8,tol=0.002)
    d=np.linalg.norm(b.obs[SL:SL+3]-p0)
    if d>0.003:
        hit=(e.copy(),y,d); break
if hit is None:
    print("G=%.1f NOCONTACT final EE %s err %.4f can %s"%(G,e,err,b.obs[SL:SL+3]))
else:
    e,y,d=hit
    print("G=%.1f CONTACT: EE=%s cmd_y=%.4f can_y=%.4f  gap_y=%.4f (EEy-cany) canmove=%.4f can=%s"%(G,e,y,can[1],e[1]-can[1],d,b.obs[SL:SL+3]))
print("steps",b.n)
b.env.close()
