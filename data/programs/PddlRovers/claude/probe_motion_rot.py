from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def opos(obs,n):
    o=obs.get_object_from_name(n); return float(obs.get(o,'x')),float(obs.get(o,'y'))
PX,PY=opos(obs,'obstacle9')
obs=setth(env,obs,0.0,0)
obs,ok=nav(env,obs,PX+0.5,PY,0)
obs=setth(env,obs,np.pi/2,0)
obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00002)
print("at th=90 clearance %.5f"%(p[0]-PX))
# now rotate toward 45 deg (needs more room): does it get rejected?
for k in range(6):
    p0=pose(obs,0)
    obs,m,d=try_move(env,obs,0,0,-0.1,0)
    p1=pose(obs,0)
    print("  rot -0.1: dtheta=%.4f  x=%.5f (theta=%.4f)"%(p1[2]-p0[2],p1[0],p1[2]))
# blocked translation + rotation coupling
obs,m,d=try_move(env,obs,-0.05,0,0.2,0); print("blocked trans(-0.05x)+rot0.2 ->",np.round(d,4))
obs,m,d=try_move(env,obs,0.05,0,0.2,0);  print("free trans(+0.05x)+rot0.2 ->",np.round(d,4))
# all-or-nothing diagonal: at pillar, dx blocked but dy free?
obs,ok=nav(env,obs,PX+0.5,PY,0); obs=setth(env,obs,0.0,0)
obs,p=push_dir(env,obs,-1,0,0,coarse=0.1,fine=0.00002)
obs,m,d=try_move(env,obs,-0.05,0.2,0,0); print("at pillar: (-0.05,+0.2) ->",np.round(d,4))
obs,m,d=try_move(env,obs,-0.05,0.0,0,0); print("at pillar: (-0.05,0) ->",np.round(d,4))
env.close()
