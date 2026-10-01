import numpy as np
from probe_vis_lib import *
A=-0.085
env=make_env()
cands=[]
for seed in range(80):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    side = 1 if O[0]>0 else -1
    for n in L:
        if not (n.startswith('obstacle') and L[n]['half_z']>0.1): continue
        q=np.array([L[n]['x'],L[n]['y']]); dqo=np.linalg.norm(q-O)
        if not (0.3<dqo<0.75): continue
        u=(q-O)/dqo
        # reachable far position at d up to 1.9
        Pf=O+u*1.9
        if abs(Pf[0])<0.45 or abs(Pf[0])>2.0 or abs(Pf[1])>2.0: continue
        if np.sign(Pf[0])!=side: continue
        bad=False
        for n2 in L:
            if n2.startswith('obstacle') and n2!=n:
                q2=np.array([L[n2]['x'],L[n2]['y']]); hx2=L[n2]['half_x']
                w=q2-Pf; uu=(O-Pf)/1.9; t=np.dot(w,uu); pe=abs(w[0]*uu[1]-w[1]*uu[0])
                if L[n2]['half_z']>0.1 and 0<t<1.9 and pe<hx2+0.12: bad=True
                if np.linalg.norm(Pf-q2)<hx2+0.28: bad=True
        if not bad: cands.append((seed,n,side,round(dqo,3)))
env.close()
west=[c for c in cands if c[2]<0][:3]; east=[c for c in cands if c[2]>0][:3]
print("west cands",west); print("east cands",east)
def run(seed,pil,i):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    q=np.array([L[pil]['x'],L[pil]['y']]); dqo=np.linalg.norm(q-O); u=(q-O)/dqo
    free,xs,res=build_grid(obs); first=True
    th = np.pi if i==0 else 0.0
    e=np.array([np.cos(th),np.sin(th)])
    print("  seed%d rover%d pillar %s dqo=%.2f"%(seed,i,pil,dqo))
    for d in [dqo+0.3, 1.0, 1.4, 1.7, 1.9]:
        if d<dqo+0.28: continue
        P=O+u*d - A*e
        if abs(P[0])<0.42 or abs(P[0])>2.05 or abs(P[1])>2.05: continue
        if first: obs,ok=nav(env,obs,P[0],P[1],i=i,tol=0.01,free=free,xs=xs,res=res); first=False
        else: obs,ok=goto(env,obs,P[0],P[1],i=i,tol=0.005)
        p=pose(obs,i)
        if np.linalg.norm(p[:2]-P)>0.03: print("    d=%.2f unreachable"%d); continue
        c=p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])]); dd=np.linalg.norm(O-c); uu=(O-c)/dd
        w=q-c; perp=abs(w[0]*uu[1]-w[1]*uu[0]); frac=np.dot(w,uu)/dd
        obs,v=vis_test(env,obs,i)
        print("    d=%.3f perp=%.4f frac=%.3f vis=%s"%(dd,perp,frac,v))
    env.close()
for seed,pil,side,dqo in west: run(seed,pil,1)
for seed,pil,side,dqo in east: run(seed,pil,0)
