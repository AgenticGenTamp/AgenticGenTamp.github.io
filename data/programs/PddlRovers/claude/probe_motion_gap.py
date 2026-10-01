from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
obs=setth(env,obs,0.0,0)
for y in [1.45,1.50,1.52,1.55,1.57,1.60,1.65]:
    obs,ok=nav(env,obs,0.45,y,0); p=pose(obs,0)
    if abs(p[0]-0.45)>0.02 or abs(p[1]-y)>0.02: print("y=%.2f navfail %s"%(y,np.round(p,3))); continue
    obs,pp=push_dir(env,obs,-1,0,0,coarse=0.05,fine=0.00005)
    print("y=%.2f xmin=%.5f"%(y,pp[0]))
    # from there try to push north
    obs,pn=push_dir(env,obs,0,1,0,coarse=0.05,fine=0.00005)
    print("     then north to y=%.5f at x=%.5f"%(pn[1],pn[0]))
    obs,ok=nav(env,obs,0.45,1.3,0)
env.close()
