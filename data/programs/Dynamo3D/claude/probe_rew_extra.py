from env_client import make_env
import numpy as np
# check info dict / reward at termination, and reward after termination
env=make_env(); obs,info=env.reset(seed=11)
R=obs.get_object_from_name("robot"); C=obs.get_object_from_name("obstacle_chair")
import math
for i in range(200):
    bx,by=obs.get(R,"pos_base_x"),obs.get(R,"pos_base_y")
    cx,cy=obs.get(C,"x"),obs.get(C,"y")
    d=math.hypot(cx-bx,cy-by)
    a=np.zeros(11,dtype=np.float32); a[0]=0.1*(cx-bx)/d; a[1]=0.1*(cy-by)/d
    obs,r,te,tr,inf=env.step(a)
    if te or tr:
        print("TERM step",i,"r",repr(float(r)),"info",inf)
        for j in range(3):
            obs,r,te,tr,inf=env.step(np.zeros(11,dtype=np.float32))
            print("  post",j,repr(float(r)),te,tr,inf)
        break
env.close()
