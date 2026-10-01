import numpy as np
from probe_vis_lib import *
env=make_env()
best=[]
for seed in range(60):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    pil=[(n,np.array([L[n]['x'],L[n]['y']])) for n in L if n.startswith('obstacle') and L[n]['half_z']>0.1]
    others=[(n,np.array([L[n]['x'],L[n]['y']]),L[n]['half_x'],L[n]['half_z']) for n in L if n.startswith('obstacle')]
    for n,q in pil:
        dqo=np.linalg.norm(q-O)
        if dqo<0.3 or dqo>1.2: continue
        u=(q-O)/dqo
        for Lb in [0.4,0.6,0.8]:
            P=q+u*Lb
            d=np.linalg.norm(P-O)
            if d>1.85: continue
            ray=(O-P)/d; ax=abs(ray[0])+abs(ray[1])
            if ax>1.22: continue   # want near axis-aligned
            nrm=np.array([-ray[1],ray[0]])
            okall=True
            for s in np.arange(-0.25,0.251,0.05):
                Ps=P+nrm*s
                if abs(Ps[0])<0.40 or abs(Ps[0])>2.0 or abs(Ps[1])>2.0: okall=False;break
                if np.sign(Ps[0])!=np.sign(O[0]) and abs(O[0])>0.4: okall=False;break
                for n2,q2,hx2,hz2 in others:
                    if np.linalg.norm(Ps-q2)<hx2+0.28: okall=False;break
                    if n2!=n:
                        dd=np.linalg.norm(O-Ps); uu=(O-Ps)/dd; w=q2-Ps
                        t=np.dot(w,uu); pp=abs(w[0]*uu[1]-w[1]*uu[0])
                        if 0<t<dd and pp<hx2+0.20 and hz2>0.1: okall=False;break
                if not okall: break
            if okall:
                best.append((seed,n,round(Lb,2),tuple(np.round(P,3)),round(float(d),3),round(float(ax),3)))
env.close()
print(len(best))
for b in best[:12]: print(b)
