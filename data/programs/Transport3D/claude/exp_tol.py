import numpy as np, kutil
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
ang=np.arctan2(cu[1],cu[0]); bp=cu[:2]-0.45*np.array([np.cos(ang),np.sin(ang)])
c.gotobase(bp[0],bp[1],ang); c.move_to([cu[0],cu[1],0.3]); c.move_to([cu[0],cu[1],cu[2]]); c.grip(True)
print("grasp",c.robot()[2])
for gap in [0.05,0.03,0.02,0.015,0.01,0.005,0.002]:
    z=0.025+gap
    ok=c.move_to([cu[0],cu[1],z])
    c.grip(False)
    print("gap",gap,"mov",ok,"grasp",c.robot()[2])
    if c.robot()[2]<0.5: break
env.close()
