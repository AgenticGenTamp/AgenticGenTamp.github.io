import numpy as np, kutil, kin, off_lib2 as L
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
c.gotobase(cu[0]+0.42,cu[1],np.pi)
b,q,g=c.robot(); yaw=b[2]; print("base",np.round(b,3),"yaw",round(yaw,3))
print("vpath HI",L.vpath(c,[cu[0],cu[1],0.14],yaw))
T=L.fkq(c); print("at HI fk",np.round(T[:3,3],4),"zax",np.round(T[:3,2],3))
for dx in [0.0,-0.015,-0.03]:
    # go up, over, down finely
    T=L.fkq(c); L.vpath(c,[T[0,3],T[1,3],0.14],yaw,step=0.03)
    L.vpath(c,[cu[0]+dx,cu[1],0.14],yaw,step=0.03)
    qd,e=L.vik(c,[cu[0]+dx,cu[1],cu[2]],yaw)
    ok=c.goto(qd)
    T2=L.fkq(c); b2,q2,_=c.robot()
    print("dx=%.3f ikerr=%.4f goto=%s fk=%s dq=%.4f"%(dx,e,ok,np.round(T2[:3,3],4),np.abs(qd-q2).max()))
    c.grip(True); print("   grasp",c.robot()[2],"cube",np.round(c.opos("cube1"),4))
    if c.robot()[2]>0.5: c.grip(False)
print("steps",c.steps)
env.close()
