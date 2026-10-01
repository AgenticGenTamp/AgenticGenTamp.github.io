from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def refine(obs,ux,uy,i):
    obs,p = push_dir(env,obs,ux,uy,i,coarse=0.2,fine=0.00002,limit=600)
    return obs,p
obs,p = refine(obs,0,-1,0); print("r0 south x=1.0 ymin", p[1])
obs,p = refine(obs,1,0,0);  print("r0 east  y=%.4f xmax"%p[1], p[0])
obs,ok=goto(env,obs,1.0,0.0,0)
obs,p = refine(obs,1,0,0);  print("r0 east  y=0 xmax", p[0])
obs,ok=goto(env,obs,1.0,1.0,0)
obs,p = refine(obs,0,1,0);  print("r0 north x=1.0 ymax", p[1])
obs,ok=goto(env,obs,-1.0,0.0,1)
obs,p = refine(obs,-1,0,1); print("r1 west  y=0 xmin", p[0])
obs,ok=goto(env,obs,-1.0,-2.0,1)
obs,p = refine(obs,0,-1,1); print("r1 south x=-1.0 ymin", p[1])
env.close()
