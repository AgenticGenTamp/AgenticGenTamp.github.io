import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
print("grasp",c.robot()[2])
# carry pose: 0.3 forward of arm base, z=0.6
b,q,g=c.robot(); ax,ay=c.armbase(b)
cp=[ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.6]
print("carry",c.move_to(cp),"cube",np.round(c.opos("cube0"),3))
# drive around to front of table
print("drive to (0.15,0,0)",c.gotobase(0.15,0.0,0.0))
b,q,g=c.robot(); print("base",np.round(b,3),"cube",np.round(c.opos("cube0"),3))
for pz in [0.6,0.5,0.4255]:
    print("mv",pz,c.move_to([0.6,-0.25,pz]),"cube",np.round(c.opos("cube0"),3))
c.grip(False)
print("released? grasp",c.robot()[2],"cube",np.round(c.opos("cube0"),3),"done",c.done,"steps",c.steps)
env.close()
