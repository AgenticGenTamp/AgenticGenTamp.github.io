from probe_motion_lib import *
import numpy as np
env, obs, info = new_env(0)
def refine(obs,ux,uy,i=0):
    n=np.hypot(ux,uy); ux,uy=ux/n,uy/n
    obs,p = push_dir(env,obs,ux,uy,i,coarse=0.15,fine=0.00002,limit=600)
    return obs,p
# theta dependence of wall limit: set theta then probe east at y=0
obs,ok=goto(env,obs,1.0,0.0,0)
for th in [0.0, 0.4, 0.8]:
    # set theta absolute-ish
    while abs(pose(obs,0)[2]-th)>0.01:
        d=th-pose(obs,0)[2]
        obs,m,dd=try_move(env,obs,0,0,float(np.clip(d,-0.4,0.4)),0)
    obs,p=refine(obs,1,0,0); print("theta=%.2f east xmax %.6f"%(pose(obs,0)[2],p[0]))
    obs,ok=goto(env,obs,1.0,0.0,0)
# pillar obstacle9 at (0.654,-0.199)
px,py=0.654,-0.199
obs,ok=goto(env,obs,px+0.6,py,0); print("goto pillar+x",ok,np.round(pose(obs,0),4))
obs,p=refine(obs,-1,0,0); print("pillar from +x: x=%.6f  gapfromcenter=%.6f"%(p[0],p[0]-px))
obs,ok=goto(env,obs,px,py-0.6,0)
obs,p=refine(obs,0,1,0); print("pillar from -y: y=%.6f  gap=%.6f"%(p[1],py-p[1]))
obs,ok=goto(env,obs,px+0.5,py-0.5,0)
obs,p=refine(obs,-1,1,0); print("pillar diag(-x,+y): pos=(%.5f,%.5f) dist=%.6f"%(p[0],p[1],np.hypot(p[0]-px,p[1]-py)))
env.close()
