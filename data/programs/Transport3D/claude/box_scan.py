import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
b,q,g=c.robot(); axb,ayb=c.armbase(b)
def T(q): return kin.fk(q,base_x=axb,base_y=ayb,base_rot=b[2],mount=(0,0,0.245))
rng=np.random.default_rng(0)
rows=[]
for k in range(70):
    off=rng.uniform(-0.08,0.08,3); off[2]=rng.uniform(-0.05,0.09)
    yaw=rng.uniform(-np.pi,np.pi)
    c.move_to([cu[0],cu[1],0.35])
    c.move_to(list(cu+off),yaw=yaw)
    _,q2,_=c.robot(); M=T(q2); p=M[:3,3]-cu
    xa=np.arctan2(M[1,0],M[0,0])
    zdown=M[2,2]
    c.grip(True); s=c.robot()[2]
    if s>0.5:
        c.grip(False)
        if c.robot()[2]>0.5:
            print("STUCK"); break
    rows.append((p,xa,zdown,s))
    print("%+.3f %+.3f %+.3f xang%+.2f zd%+.2f %d"%(p[0],p[1],p[2],xa,zdown,int(s)))
np.save("scan1.npy",np.array([[*r[0],r[1],r[2],r[3]] for r in rows]))
print("steps",c.steps,"cubemoved",np.round(c.opos("cube0")-cu,4))
env.close()
