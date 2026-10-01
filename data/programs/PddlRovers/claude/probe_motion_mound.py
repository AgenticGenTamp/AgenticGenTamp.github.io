from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
obs=setth(env,obs,0.0,0); obs=setth(env,obs,0.0,1)
def opos(obs,n):
    o=obs.get_object_from_name(n); return float(obs.get(o,'x')),float(obs.get(o,'y'))
MX,MY=opos(obs,'obstacle0'); print("mound0",MX,MY)
# south approach at x=MX
obs,ok=nav(env,obs,MX,1.0,0); print("nav",ok,np.round(pose(obs,0),3))
obs,p=push_dir(env,obs,0,1,0,coarse=0.1,fine=0.00005); print("mound from -y: y=%.5f  clearance_from_face=%.5f"%(p[1],MY-0.25-p[1]))
# west approach at y=MY
obs,ok=nav(env,obs,1.0,MY,0); print("nav",ok,np.round(pose(obs,0),3))
obs,p=push_dir(env,obs,1,0,0,coarse=0.1,fine=0.00005); print("mound from -x: x=%.5f  clearance=%.5f"%(p[0],MX-0.25-p[0]))
# diagonal 45 toward corner (MX-0.25,MY-0.25)
obs,ok=nav(env,obs,1.2,1.2,0); print("nav",ok,np.round(pose(obs,0),3))
obs,p=push_dir(env,obs,0.7071,0.7071,0,coarse=0.1,fine=0.00005)
cx,cy=MX-0.25,MY-0.25
print("diag stop (%.4f,%.4f) dist_to_corner=%.5f"%(p[0],p[1],np.hypot(p[0]-cx,p[1]-cy)))
for (dx,dy) in [(0.01,0),(0,0.01),(0.01,0.01)]:
    obs,m,d=try_move(env,obs,dx,dy,0,0); print("  probe",dx,dy,"moved",m)
    if m: obs,_,_=try_move(env,obs,-dx,-dy,0,0)
# drive-over test: try to go to mound center
obs,ok=nav(env,obs,MX,MY,0,maxit=200); print("nav to mound center reached?",ok,np.round(pose(obs,0),3))
# mound theta dependence: support at theta=45
obs,ok=nav(env,obs,MX,1.0,0)
obs=setth(env,obs,np.pi/4,0)
obs,p=push_dir(env,obs,0,1,0,coarse=0.1,fine=0.00005); print("mound from -y at th=45: clearance=%.5f"%(MY-0.25-p[1]))
# wall theta dependence
obs=setth(env,obs,0.0,0)
obs,ok=nav(env,obs,1.0,-1.0,0)
obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005); print("midwall th=0 xmin=%.5f"%p[0])
obs=setth(env,obs,np.pi/4,0)
obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00005); print("midwall th=45 xmin=%.5f"%p[0])
obs,p=push_dir(env,obs,0,-1,0,coarse=0.1,fine=0.00005); print("south wall th=45 ymin=%.5f"%p[1])
env.close()
