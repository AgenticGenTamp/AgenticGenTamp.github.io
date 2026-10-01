import numpy as np, sys
from probe_vis_lib import *
A=-0.085
def run(SEED,PIL,i,d0):
    env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    q=np.array([L[PIL]['x'],L[PIL]['y']]); dqo=np.linalg.norm(q-O); u=(q-O)/dqo
    free,xs,res=build_grid(obs)
    th=np.pi if i==0 else 0.0
    e=np.array([np.cos(th),np.sin(th)]); nrm=np.array([-u[1],u[0]])
    print("== seed%d rover%d pillar%s q=%s O=%s dqo=%.2f"%(SEED,i,PIL,np.round(q,3),np.round(O,3),dqo))
    first=True; out=[]
    for s in np.arange(-0.36,0.361,0.03):
        P=O+u*d0+nrm*s-A*e
        if abs(P[0])<0.42 or abs(P[0])>2.05 or abs(P[1])>2.05: continue
        if first: obs,ok=nav(env,obs,P[0],P[1],i=i,tol=0.01,free=free,xs=xs,res=res); first=False
        else: obs,ok=goto(env,obs,P[0],P[1],i=i,tol=0.005)
        p=pose(obs,i)
        if np.linalg.norm(p[:2]-P)>0.03: out.append((s,None,None)); continue
        c=p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])]); dd=np.linalg.norm(O-c); uu=(O-c)/dd
        w=q-c; perp=w[0]*uu[1]-w[1]*uu[0]
        obs,v=vis_test(env,obs,i)
        out.append((round(float(s),3),round(float(perp),4),v))
    for s,pe,v in out:
        print("   s=%+.2f signed_perp=%s %s"%(s,pe,'#' if v else ('.' if v is not None else '?')))
    env.close()
run(20,'obstacle7',0,1.4)
run(23,'obstacle5',0,1.4)
