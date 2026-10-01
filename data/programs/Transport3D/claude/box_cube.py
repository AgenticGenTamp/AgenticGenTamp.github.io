import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,_=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
b,_,_=c.robot(); ax,ay=c.armbase(b)
c.move_to([ax+0.30*np.cos(b[2]), ay+0.30*np.sin(b[2]), 0.6])
c.gotobase(0.15,0.0,0.0)
for pz in [0.6,0.5,0.44]: c.move_to([0.6,-0.3,pz])
for tol,gt in [(None,1e-4),(1e-4,1e-5),(1e-5,1e-6),(1e-6,1e-6)]:
    r=c.move_to([0.6,-0.3,0.425],tol=tol,gtol=gt)
    print("tol",tol,r,"cube",np.round(c.opos("cube0"),6))
c.grip(False); print("rel",c.robot()[2],np.round(c.opos("cube0"),6),"done",c.done,"steps",c.steps)
env.close()
