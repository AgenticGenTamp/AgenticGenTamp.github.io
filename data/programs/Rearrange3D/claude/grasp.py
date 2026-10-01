import sys, numpy as np
from env_client import make_env
from ctrl import Bot
np.set_printoptions(precision=4,suppress=True)
OPENV=float(sys.argv[1]); CLOSEV=1.0-OPENV
Z=float(sys.argv[2])
b=Bot(make_env(),0); SL=32; can=b.obs[SL:SL+3].copy()
b.grip(OPENV,5)
y0=can[1]+0.17
b.servo(np.array([can[0],y0,0.70]),OPENV,200,tol=0.004)
b.servo(np.array([can[0],y0,Z]),OPENV,140,tol=0.004)
for ycmd in [-0.30,-0.34,-0.38,-0.42]:
    e,err=b.servo(np.array([can[0],ycmd,Z]),OPENV,45,tol=0.002)
    pre=b.obs[SL:SL+3].copy()
    b.grip(CLOSEV,14)
    dclose=np.linalg.norm(b.obs[SL:SL+3]-pre)
    z2=b.obs[SL+2]
    e2,_=b.servo(np.array([e[0],e[1],Z+0.20]),CLOSEV,90,tol=0.005)
    dz=b.obs[SL+2]-z2
    print("open=%.1f ycmd=%.2f EE=%s | close_move=%.4f LIFT_dz=%+.4f can=%s"%(OPENV,ycmd,e,dclose,dz,b.obs[SL:SL+3]))
    if dz>0.03:
        bx=b.obs[93]; p=b.obs[SL:SL+3].copy()
        a=np.zeros(11); a[0]=-0.1; a[10]=CLOSEV
        for _ in range(20): b.step(a)
        print("*** GRASPED. base dx=%.4f can d=%s"%(b.obs[93]-bx, b.obs[SL:SL+3]-p))
        break
    b.grip(OPENV,8)
    b.servo(np.array([e[0],e[1],Z]),OPENV,40,tol=0.005)
print("steps",b.n)
b.env.close()
