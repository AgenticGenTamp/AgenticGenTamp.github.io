import numpy as np, kutil, kin
from env_client import make_env
kutil.MZ=0.269
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
cu=c.opos("cube1")
# stand at distance 0.45 in a fixed rot=0, cube behind/left/right
for ang_off in [0.0, np.pi/2, np.pi, -np.pi/2]:
    u=np.array([np.cos(ang_off),np.sin(ang_off)])
    bpos=cu[:2]-0.45*u
    ok0=c.gotobase(bpos[0],bpos[1],0.0)
    r1=c.move_to([cu[0],cu[1],0.35], yaw=ang_off)
    r2=c.move_to([cu[0],cu[1],cu[2]], yaw=ang_off)
    c.grip(True); g=c.robot()[2]
    print("approach dir",round(ang_off,2),"drive",ok0,"pre",r1,"desc",r2,"grasp",g,"steps",c.steps)
    if g>0.5:
        c.move_to([cu[0],cu[1],0.3],yaw=ang_off); c.move_to([cu[0],cu[1],cu[2]],yaw=ang_off); c.grip(False)
        print("   released",c.robot()[2])
env.close()
