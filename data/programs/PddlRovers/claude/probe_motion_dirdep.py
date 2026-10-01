from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
px,py=0.654,-0.199
def setth(obs,th,i=0):
    while abs(pose(obs,i)[2]-th)>0.005:
        d=th-pose(obs,i)[2]
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),i)
    return obs
obs=setth(obs,0.0)
# 1) approach the point rel=(0.2016,0.2016) from -y along column rel_x=0.2016
obs,ok=goto(env,obs,px+0.2016, py-0.6,0); print("start",ok,np.round(pose(obs,0),4))
obs,p=push_dir(env,obs,0,1,0,coarse=0.1,fine=0.00005); print("north along relx=0.2016 -> rely=%.4f"%(p[1]-py))
# 2) approach from +y
obs,ok=goto(env,obs,px+0.2016, py+0.6,0)
obs,p=push_dir(env,obs,0,-1,0,coarse=0.1,fine=0.00005); print("south along relx=0.2016 -> rely=%.4f"%(p[1]-py))
# 3) pure 45 deg again at theta=0
obs,ok=goto(env,obs,px+0.5,py+0.5,0)
obs,p=push_dir(env,obs,-0.7071,-0.7071,0,coarse=0.1,fine=0.00005); print("45deg approach rel=(%.4f,%.4f)"%(p[0]-px,p[1]-py))
# 4) from that blocked pos, can it move purely +? try small pure -x and -y steps
for (dx,dy) in [(-0.01,0),(0,-0.01),(-0.01,-0.01),(0.01,0),(0,0.01)]:
    obs,m,d=try_move(env,obs,dx,dy,0,0); print("  probe",dx,dy,"moved",m)
    if m: obs,m2,d2=try_move(env,obs,-dx,-dy,0,0)
print("pos",np.round(pose(obs,0),4))
env.close()
