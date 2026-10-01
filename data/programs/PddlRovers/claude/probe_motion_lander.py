from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
obs=setth(env,obs,0.0,1)
def opos(obs,n):
    o=obs.get_object_from_name(n); return float(obs.get(o,'x')),float(obs.get(o,'y'))
LX,LY=opos(obs,'lander'); print("lander",LX,LY)
# from +x at y=LY
obs,ok=nav(env,obs,-1.0,LY,1); print("nav",ok,np.round(pose(obs,1),3))
obs,p=push_dir(env,obs,-1,0,1,coarse=0.1,fine=0.00005); print("lander from +x: x=%.5f  dx_from_center=%.5f"%(p[0],p[0]-LX))
# from +y at x=-2.2 and x=-1.6
for xc in [-2.25,-2.0,-1.9,-1.6,-1.4]:
    obs,ok=nav(env,obs,xc,-1.2,1)
    p0=pose(obs,1)
    if abs(p0[0]-xc)>0.02 or abs(p0[1]+1.2)>0.02: print("x=%.2f navfail %s"%(xc,np.round(p0,3))); continue
    obs,p=push_dir(env,obs,0,-1,1,coarse=0.1,fine=0.00005); print("x=%5.2f from +y: y=%.5f dy_from_center=%.5f"%(xc,p[1],p[1]-LY))
    obs,ok=nav(env,obs,xc,-1.2,1)
# theta dependence of lander
obs,ok=nav(env,obs,-1.0,LY,1)
obs=setth(env,obs,np.pi/4,1)
obs,p=push_dir(env,obs,-1,0,1,coarse=0.1,fine=0.00005); print("lander from +x th=45: dx=%.5f"%(p[0]-LX))
env.close()
