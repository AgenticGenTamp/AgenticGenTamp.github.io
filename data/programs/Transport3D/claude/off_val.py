import sys, numpy as np, kutil, kin, off_lib2 as L
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
seed=int(sys.argv[1]); TZ=float(sys.argv[2]); tag=sys.argv[3]
env=make_env(); o,info=env.reset(seed=seed)
c=kutil.Ctl(env,o)
names=sorted(n for n in c.o.get_object_names() if n.startswith("cube"))
res=[]
for i,nm in enumerate(names[:4]):
    cu=c.opos(nm)
    if cu[2]>0.1: continue
    th=[0.0,0.8,-0.9,2.2][i%4]; D=[0.42,0.50,0.36,0.46][i%4]
    if not c.gotobase(cu[0]+D*np.cos(th),cu[1]+D*np.sin(th),th+np.pi): res.append((nm,"basefail")); continue
    b,q,g=c.robot(); yaw=b[2]+[0,1.5708,3.1416,-1.5708][i%4]
    HI=0.16
    def goover(x,y):
        T=L.fkq(c)
        if L.vpath(c,[T[0,3],T[1,3],HI],yaw,step=0.04) and L.vpath(c,[x,y,HI],yaw,step=0.04): return True
        ok,e=L.vmove(c,[x,y,HI],yaw); return ok or L.vpath(c,[x,y,HI],yaw,step=0.05)
    if not goover(cu[0],cu[1]): res.append((nm,"pregrasp")); continue
    if not L.vpath(c,[cu[0],cu[1],cu[2]+TZ],yaw,step=0.03): res.append((nm,"descend")); continue
    T=L.fkq(c); c.grip(True); s=c.robot()[2]
    off=T[:3,:3].T@(cu-T[:3,3])
    if s>0.5:
        L.vpath(c,[cu[0],cu[1],0.25],yaw,step=0.04)
        lifted=c.opos(nm)[2]
        res.append((nm,"GRASP z=%.3f offtool=%s"%(lifted,np.round(off,4))))
        L.vpath(c,[cu[0],cu[1],cu[2]+TZ],yaw,step=0.04); c.grip(False)
    else: res.append((nm,"nograsp offtool=%s"%np.round(off,4)))
    if c.steps>850: break
print("### seed=%d TZ=%.3f steps=%d"%(seed,TZ,c.steps))
for r in res: print("  ",r[0],r[1])
env.close()
