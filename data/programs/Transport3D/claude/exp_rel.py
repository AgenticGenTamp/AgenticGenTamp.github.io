import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=1)
c=kutil.Ctl(env,o)
cu=c.opos("cube0")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang)
c.move_to([cu[0],cu[1],0.3]); print("desc",c.move_to([cu[0],cu[1],cu[2]]))
c.grip(True); print("grasp",c.robot()[2], "cube", np.round(c.opos("cube0"),3))
c.move_to([cu[0],cu[1],0.45])
# release in free space above floor at various heights
for z in [0.4,0.2,0.1,0.05,0.026,0.025]:
    ok=c.move_to([cu[0],cu[1],z])
    c.grip(False)
    print("rel at",z,"movok",ok,"grasp",c.robot()[2],"cube",np.round(c.opos("cube0"),3))
    if c.robot()[2]<0.5: break
env.close()
