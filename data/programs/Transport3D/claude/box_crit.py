import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
print("cube",np.round(cu,3),"baserot",round(c.robot()[0][2],3))
res=[]
def trial(off,yaw):
    p=cu+np.array(off)
    ok=c.move_to(list(p),yaw=yaw)
    if not ok: return "X"
    c.grip(True); g=c.robot()[2]
    if g>0.5:
        c.grip(False); g2=c.robot()[2]
        cu2=c.opos("cube0")
        return "G" if g2<0.5 and np.allclose(cu2,cu,atol=1e-3) else "G!"
    return "."
# axis sweeps
for yaw in [0.0, np.pi/4, np.pi/2]:
    for ax,i in [("dx",0),("dy",1),("dz",2)]:
        row=[]
        for d in [-0.08,-0.06,-0.05,-0.04,-0.03,-0.02,0.0,0.02,0.03,0.04,0.05,0.06,0.08]:
            off=[0,0,0]; off[i]=d
            row.append(trial(off,yaw))
        print("yaw%.2f %s"%(yaw,ax), "".join(row), "steps",c.steps)
env.close()
