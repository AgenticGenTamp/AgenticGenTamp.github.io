import numpy as np
from probe_vis_lib import *
A=-0.085
def camp(obs,i):
    p=pose(obs,i); return p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])])
for SEED in [10,41]:
    env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    free,xs,res=build_grid(obs)
    print("== seed",SEED,"O",np.round(O,3))
    # rover0 east positions at various distance, ray crosses x=0
    for P in [(0.35,1.60),(0.40,1.20),(0.60,1.40),(0.90,1.60),(1.20,1.80)]:
        obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.02,free=free,xs=xs,res=res)
        p=pose(obs,0)
        if np.linalg.norm(p[:2]-np.array(P))>0.05: print("  r0 %s unreachable (%s)"%(P,np.round(p[:2],2))); continue
        c=camp(obs,0); d=np.linalg.norm(O-c); uu=(O-c)/d
        tcross=(0-c[0])/uu[0] if abs(uu[0])>1e-6 else -1
        ycross=c[1]+tcross*uu[1] if tcross>0 else None
        # pillars near ray
        pil=[]
        for n in L:
            if n.startswith('obstacle') and L[n]['half_z']>0.1:
                q=np.array([L[n]['x'],L[n]['y']]); w=q-c; t=np.dot(w,uu); pe=abs(w[0]*uu[1]-w[1]*uu[0])
                if 0<t<d and pe<0.12: pil.append((n,round(pe,3),round(t/d,2)))
        obs,v=vis_test(env,obs,0)
        print("  r0 at %s d=%.3f vis=%s  wall_cross_y=%s pillars=%s"%(np.round(p[:2],2),d,v,None if ycross is None else round(ycross,2),pil))
    # control rover1 west
    for P in [(-0.40,1.20),(-0.60,0.60)]:
        obs,ok=nav(env,obs,P[0],P[1],i=1,tol=0.02,free=free,xs=xs,res=res)
        p=pose(obs,1)
        if np.linalg.norm(p[:2]-np.array(P))>0.05: print("  r1 %s unreachable"%(P,)); continue
        c=camp(obs,1); d=np.linalg.norm(O-c)
        obs,v=vis_test(env,obs,1)
        print("  r1 at %s d=%.3f vis=%s (no wall crossing)"%(np.round(p[:2],2),d,v))
    env.close()
