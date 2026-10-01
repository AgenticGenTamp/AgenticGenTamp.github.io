from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
# clipping vs rejection at wall
obs,ok=goto(env,obs,2.0,0.0,0)
obs,m,d=try_move(env,obs,0.2,0,0,0); print("from2.0 +0.2 ->",np.round(pose(obs,0)[:2],5))
obs,m,d=try_move(env,obs,0.2,0,0,0); print("again   +0.2 ->",np.round(pose(obs,0)[:2],5), "moved",m)
px,py=0.654,-0.199
def radial(obs, ang_deg, i=0, start=0.65):
    a=np.deg2rad(ang_deg); ux,uy=np.cos(a),np.sin(a)
    obs,ok=goto(env,obs,px+start*ux,py+start*uy,i)
    if not ok: return obs,None
    obs,p=push_dir(env,obs,-ux,-uy,i,coarse=0.1,fine=0.00005,limit=600)
    rel=np.array([p[0]-px,p[1]-py])
    return obs,(ang_deg, rel[0],rel[1], np.hypot(*rel), pose(obs,i)[2])
for a in [0,15,30,45,60,75,90,135,180,225,270]:
    obs,r = radial(obs,a)
    if r: print("ang %4d rel=(%.4f,%.4f) d=%.4f th=%.2f"%r)
    else: print("ang",a,"unreachable")
env.close()
