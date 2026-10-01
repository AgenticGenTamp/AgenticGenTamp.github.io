import numpy as np, kutil
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang); c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
c.move_to([cu[0],cu[1],0.65])
b,q,g=c.robot(); ax,ay=c.armbase(b)
c.move_to([ax+0.3*np.cos(b[2]),ay+0.3*np.sin(b[2]),0.65])
c.gotobase(0.17,0.0,0.0)
c.move_to([0.6,0.0,0.65])
for z in [0.4250,0.4255,0.426]:
    ok=c.move_to([0.6,0.0,z])
    print("move",z,ok,"ee cube",np.round(c.opos("cube0"),4))
    if ok: break
c.grip(False)
print("grasp",c.robot()[2],"cube",np.round(c.opos("cube0"),4),"done",c.done)
for i in range(3):
    t,tr=c.step(np.zeros(11)); print("noop term",t)
env.close()
