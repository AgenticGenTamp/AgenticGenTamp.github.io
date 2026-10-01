import sys, numpy as np
from env_client import make_env
from ctrl import Bot
np.set_printoptions(precision=4,suppress=True)
G=float(sys.argv[1]); Z=float(sys.argv[2])
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy()
b.grip(G,5)
y0=can[1]+0.17
e,err=b.servo(np.array([can[0],y0,0.70]),G,200,tol=0.004)
e,err=b.servo(np.array([can[0],y0,Z]),G,140,tol=0.004)
print("G=%.1f Z=%.3f start EE %s err %.4f"%(G,Z,e,err))
p0=b.obs[SL:SL+3].copy(); rows=[]; hit=None
for k in range(1,56):
    y=y0-0.005*k
    e,err=b.servo(np.array([can[0],y,Z]),G,8,tol=0.002)
    d=np.linalg.norm(b.obs[SL:SL+3]-p0)
    rows.append((y,e[0],e[1],e[2],err,d))
    if d>0.0008 and hit is None:
        hit=(e.copy(),y,d); print("G=%.1f FIRST CONTACT EE=%s (EEy-cany)=%+.4f canmove=%.4f"%(G,e,e[1]-can[1],d)); break
if hit is None:
    print("G=%.1f NOCONTACT lastEE %s err %.4f can %s"%(G,e,err,b.obs[SL:SL+3]))
    for r in rows[::8]: print("   y=%.3f EE=(%.3f,%.3f,%.3f) err=%.3f d=%.4f"%r)
print("steps",b.n)
b.env.close()
