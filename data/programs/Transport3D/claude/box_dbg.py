import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
b,q,g=c.robot(); ax,ay=c.armbase(b)
def fkp(q): return kin.fk(q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,kutil.MZ))[:3,3]
safe=[cu[0],cu[1],0.35]
for d in [0.0,0.03,-0.03]:
    for yaw in [0.0,np.pi/2]:
        ok0=c.move_to(safe,yaw=yaw)
        p=cu+np.array([d,0,0])
        qd=c.ik(list(p),yaw=yaw)
        _,q1,_=c.robot()
        ok=c.goto(qd)
        _,q2,_=c.robot()
        print("dx",d,"yaw",round(yaw,2),"safe",ok0,"goto",ok,"errmax",round(float(np.abs(qd-q2).max()),3),
              "fk_target",np.round(fkp(qd),3),"fk_now",np.round(fkp(q2),3),"dq",np.round(np.abs(qd-q1).max(),2))
env.close()
