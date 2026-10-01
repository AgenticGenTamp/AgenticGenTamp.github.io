import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=22, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
q=np.array([L['obstacle10']['x'],L['obstacle10']['y']])
print("O",np.round(O,3),"pillar",np.round(q,3))
free,xs,res=build_grid(obs)
A=-0.087
def cam(obs,i=1):
    p=pose(obs,i); return p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])])
P0=np.array([-1.705,0.52]); d0=np.linalg.norm(O-P0); ray=(O-P0)/d0
nrm=np.array([-ray[1],ray[0]])
print("ray dir",np.round(ray,3),"ax",abs(ray[0])+abs(ray[1]))
obs,ok=nav(env,obs,*(P0+nrm*(-0.2)),i=1,tol=0.01,free=free,xs=xs,res=res)
prev=None
for s in np.arange(-0.20,0.201,0.02):
    P=P0+nrm*s
    obs,_=goto(env,obs,P[0],P[1],i=1,tol=0.004)
    p=pose(obs,1)
    if np.linalg.norm(p[:2]-P)>0.02: print("  s=%.2f unreachable"%s); continue
    c=cam(obs,1); d=np.linalg.norm(O-c); uu=(O-c)/d; w=q-c
    pe=abs(w[0]*uu[1]-w[1]*uu[0]); frac=np.dot(w,uu)/d
    obs,v=vis_test(env,obs,1)
    if prev is not None and prev[0]!=v:
        print("  TRANSITION perp %.4f(%s) -> %.4f(%s)"%(prev[1],'vis' if prev[0] else 'blk',pe,'vis' if v else 'blk'))
    print("  s=%+.2f theta=%.2f perp=%.4f frac=%.2f d=%.3f %s"%(s,p[2],pe,frac,d,'#' if v else '.'))
    prev=(v,pe)
env.close()
