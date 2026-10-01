import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
b0=c.robot()[0]
safe=[cu[0],cu[1],0.35]
def trial(off,yaw):
    c.move_to(safe,yaw=yaw)
    p=cu+np.array(off)
    qd=c.ik(list(p),yaw=yaw)
    if qd is None: return "I"   # ik fail
    if not c.goto(qd): return "B"  # blocked
    # verify fk
    c.grip(True); g=c.robot()[2]
    if g>0.5:
        c.grip(False); g2=c.robot()[2]
        return "G" if g2<0.5 else "S"
    return "."
for yaw in [0.0, np.pi/2]:
    for ax,i in [("dx",0),("dy",1),("dz",2)]:
        row=[]
        for d in [-0.06,-0.05,-0.04,-0.03,-0.02,0.0,0.02,0.03,0.04,0.05,0.06]:
            off=[0,0,0]; off[i]=d
            row.append(trial(off,yaw))
        print("yaw%.2f %s"%(yaw,ax), "".join(row), "st",c.steps, "cube",np.round(c.opos("cube0")-cu,3))
env.close()
