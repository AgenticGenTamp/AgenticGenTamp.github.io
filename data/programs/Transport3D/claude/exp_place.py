import numpy as np, kutil, kin
from env_client import make_env
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0"); print("cube",np.round(cu,3),"box",np.round(c.opos("box0"),3))
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
print("drive",c.gotobase(bp[0],bp[1],ang))
print("pre",c.move_to([cu[0],cu[1],0.35]))
print("desc",c.move_to([cu[0],cu[1],0.05]))
c.grip(True); b,q,g=c.robot(); print("grasp",g)
print("lift",c.move_to([cu[0],cu[1],0.45]))
# drive to table: base in front of table at x=0.2,y=0 facing +x
print("drive2",c.gotobase(0.15,0.0,0.0))
print("cube carried",np.round(c.opos("cube0"),3))
# place at (0.55,-0.2) center z 0.425+
for pz in [0.5,0.45,0.43,0.425]:
    ok=c.move_to([0.55,-0.2,pz])
    print("place move",pz,ok,"cube",np.round(c.opos("cube0"),3))
    if ok: break
c.grip(False)
b,q,g=c.robot(); print("after release grasp",g,"cube",np.round(c.opos("cube0"),3),"done",c.done)
# retract arm up
print("retract",c.move_to([0.55,-0.2,0.6]))
print("cube after retract",np.round(c.opos("cube0"),3),"done",c.done,"steps",c.steps)
env.close()
