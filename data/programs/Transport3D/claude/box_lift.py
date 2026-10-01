import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=0)
c=kutil.Ctl(env,o)
B=c.opos("box0"); H=np.array([0.1,0.15,0.1])
tx,ty=B[0],B[1]+H[1]
print("drive",c.gotobase(tx, ty+0.45, -np.pi/2))
b,q,g=c.robot(); print("base",np.round(b,3))
print("hi",c.move_to([tx,ty,0.45]))
for z in [0.35,0.28,0.24,0.22,0.20]:
    r=c.move_to([tx,ty,z])
    if not r: print("blocked at",z); break
c.grip(True); print("grasp",c.robot()[2],"box",np.round(c.opos("box0"),3))
for z in [0.25,0.35,0.5]:
    print("lift",z,c.move_to([tx,ty,z]),"box",np.round(c.opos("box0"),3))
# rotate base / drive
print("drive2",c.gotobase(0.0,0.0,0.0),"box",np.round(c.opos("box0"),3))
b,q,g=c.robot(); print("base",np.round(b,3),"grasp",g)
env.close()
