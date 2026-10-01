import numpy as np
from probe_vis_lib import *
A=-0.085
env=make_env()
cfgs=[]
for seed in range(80):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    mounds={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('obstacle') and L[n]['half_z']<0.1}
    mown=min(mounds,key=lambda n:np.linalg.norm(mounds[n]-O))
    for n in L:
        if not (n.startswith('obstacle') and L[n]['half_z']>0.1): continue
        q=np.array([L[n]['x'],L[n]['y']]); dqo=np.linalg.norm(q-O)
        if not (0.35<dqo<1.1): continue
        u=(q-O)/dqo
        for d in [dqo+0.45, dqo+0.8]:
            if d>1.9: continue
            i=0 if O[0]>0 else 1
            th=np.pi if i==0 else 0.0
            P=O+u*d-A*np.array([np.cos(th),np.sin(th)])
            if abs(P[0])<0.42 or abs(P[0])>2.0 or abs(P[1])>2.0: continue
            if np.sign(P[0])!=np.sign(O[0]): continue
            bad=False
            for n2 in L:
                if n2.startswith('obstacle') and n2!=n:
                    q2=np.array([L[n2]['x'],L[n2]['y']]); hx2=L[n2]['half_x']
                    if np.linalg.norm(P-q2)<hx2+0.28: bad=True
                    if L[n2]['half_z']>0.1:
                        uu=(O-P)/np.linalg.norm(O-P); w=q2-P; t=np.dot(w,uu); pe=abs(w[0]*uu[1]-w[1]*uu[0])
                        if 0<t<np.linalg.norm(O-P) and pe<0.15: bad=True
            if not bad:
                cfgs.append((seed,n,i,float(d),tuple(O),tuple(q),mown,float(dqo),
                             float(np.linalg.norm(mounds[mown]-q))))
                break
env.close()
# unique per seed, up to 14
seen=set(); sel=[]
for c in cfgs:
    if c[0] in seen: continue
    seen.add(c[0]); sel.append(c)
    if len(sel)>=14: break
print("testing",len(sel),"configs")
for seed,pil,i,d,O,q,mown,dqo,dqm in sel:
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); Oa=np.array(O); qa=np.array(q); u=(qa-Oa)/dqo
    free,xs,res=build_grid(obs)
    th=np.pi if i==0 else 0.0
    P=Oa+u*d-A*np.array([np.cos(th),np.sin(th)])
    obs,ok=nav(env,obs,P[0],P[1],i=i,tol=0.01,free=free,xs=xs,res=res)
    p=pose(obs,i)
    if np.linalg.norm(p[:2]-P)>0.04:
        print("seed%-3d %-11s UNREACHED"%(seed,pil)); env.close(); continue
    c=p[:2]+A*np.array([np.cos(p[2]),np.sin(p[2])]); dd=np.linalg.norm(Oa-c); uu=(Oa-c)/dd
    w=qa-c; perp=abs(w[0]*uu[1]-w[1]*uu[0]); frac=np.dot(w,uu)/dd
    obs,_,_,_,_=st(env,op='calibrate',i=i); v=rf(obs,i)['calibrated']>0.5
    print("seed%-3d %-11s r%d O=(%.2f,%.2f) mound=%s q=(%.2f,%.2f) dqo=%.2f d=%.2f perp=%.3f frac=%.2f q-to-mound=%.2f VIS=%s"%(
        seed,pil,i,Oa[0],Oa[1],mown[-1],qa[0],qa[1],dqo,dd,perp,frac,dqm,v))
    env.close()
