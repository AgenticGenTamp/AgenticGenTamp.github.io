from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def opos(obs,n):
    o=obs.get_object_from_name(n); return float(obs.get(o,'x')),float(obs.get(o,'y'))
obs=setth(env,obs,0.0,0); obs=setth(env,obs,0.0,1)
# --- sample drive-over (rover1, sample3)
SX,SY=opos(obs,'sample3'); print("sample3",round(SX,4),round(SY,4))
obs,ok=nav(env,obs,SX,SY,1); print("nav onto sample3:",ok,np.round(pose(obs,1),4))
S4=opos(obs,'sample4'); print("sample4",np.round(S4,4))
obs,ok=nav(env,obs,S4[0],S4[1],1); print("nav onto sample4:",ok,np.round(pose(obs,1),4))
# --- rover-rover across the wall at same y, both theta=45
obs=setth(env,obs,np.pi/4,0); obs=setth(env,obs,np.pi/4,1)
obs,ok=nav(env,obs,1.0,-1.0,0); obs,ok2=nav(env,obs,-1.0,-1.0,1)
print("nav",ok,ok2,np.round(pose(obs,0),3),np.round(pose(obs,1),3))
obs,p0=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005); print("r0 xmin(th45)=%.5f"%p0[0])
obs,p1=push_dir(env,obs,1,0,1,coarse=0.1,fine=0.00005); print("r1 xmax(th45)=%.5f  sep=%.5f"%(p1[0],p0[0]-p1[0]))
# now rover1 slides in y toward rover0's y? they are at same y already
# --- pillar theta sweep fine (rover0, pillar obstacle9) 0..345
PX,PY=opos(obs,'obstacle9')
for deg in [0,30,60,90,120,150,180,210,240,270,300,330]:
    obs,ok=nav(env,obs,PX+0.5,PY,0)
    p=pose(obs,0)
    if abs(p[0]-PX-0.5)>0.02: print("deg",deg,"navfail"); continue
    obs=setth(env,obs,np.deg2rad(deg) if deg<=180 else np.deg2rad(deg-360),0)
    obs,pp=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00002)
    print("pillar th=%3d clearance_from_center=%.5f"%(deg,pp[0]-PX))
env.close()
