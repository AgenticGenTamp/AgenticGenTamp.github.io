import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
free,xs,res=build_grid(obs)
A=-0.087
def cam(obs,i=0):
    p=pose(obs,i); return p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])])
def perp_to(c,q):
    d=np.linalg.norm(O-c); uu=(O-c)/d; w=q-c
    return abs(w[0]*uu[1]-w[1]*uu[0]), np.dot(w,uu)/d, d
def sweep(obs,y,x0,x1,step,pilname):
    q=np.array([L[pilname]['x'],L[pilname]['y']])
    obs,ok=nav(env,obs,x0,y,i=0,tol=0.01,free=free,xs=xs,res=res)
    print("--- sweep y=%.2f pillar %s at %s"%(y,pilname,np.round(q,3)))
    prev=None
    for x in np.arange(x0,x1+1e-9,step):
        obs,_=goto(env,obs,x,y,i=0,tol=0.004)
        p=pose(obs,0)
        if abs(p[0]-x)>0.02: print("   x=%.2f unreachable"%x); continue
        c=cam(obs); pe,frac,d=perp_to(c,q)
        obs,v=vis_test(env,obs,0)
        mark='#' if v else '.'
        if prev is not None and prev[0]!=v:
            print("   TRANSITION between perp=%.4f (%s) and perp=%.4f (%s)"%(prev[1],'vis' if prev[0] else 'blk',pe,'vis' if v else 'blk'))
        print("   x=%.2f theta=%.2f perp=%.4f frac=%.2f d=%.3f %s"%(x,p[2],pe,frac,d,mark))
        prev=(v,pe)
    return obs
obs=sweep(obs,1.20,1.32,1.52,0.02,'obstacle5')
obs=sweep(obs,0.60,1.06,1.26,0.02,'obstacle9')
env.close()
