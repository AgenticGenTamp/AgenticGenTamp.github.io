from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
# rover0 at (1,-1.75). Drive south (clear region around x=1,y<-1.75?)
obs,p = push_dir(env,obs,0,-1,0); print("south from x=1.0: ", np.round(p,5))
# blocked: check dtheta still applies
p0 = pose(obs,0)
obs,m,d = try_move(env,obs,0.0,-0.2,0.3,0); print("blocked trans + dth:", np.round(d,5))
obs,m,d = try_move(env,obs,0.0,0.0,0.3,0); print("rot only:", np.round(d,5))
# east
obs,p = push_dir(env,obs,1,0,0); print("east at y=%.3f: "%p[1], np.round(p,5))
# north along x=2.3?
obs,ok = goto(env,obs,2.3,0.0,0); print("goto(2.3,0) ok",ok, np.round(pose(obs,0),4))
obs,p = push_dir(env,obs,1,0,0); print("east at y=0: ", np.round(p,5))
obs,ok = goto(env,obs,2.0,0.0,0)
obs,p = push_dir(env,obs,0,1,0); print("north at x=2.0: ", np.round(p,5))
# rover1 west/south
obs,ok = goto(env,obs,-1.0,-2.0,1); print("r1 goto ok",ok,np.round(pose(obs,1),4))
obs,p = push_dir(env,obs,0,-1,1); print("r1 south at x=-1.0: ", np.round(p,5))
obs,ok = goto(env,obs,-1.0,-1.0,1)
obs,p = push_dir(env,obs,-1,0,1); print("r1 west at y=-1.0: ", np.round(p,5))
env.close()
