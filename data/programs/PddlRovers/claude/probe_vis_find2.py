import numpy as np
from probe_vis_lib import *
env=make_env()
def seg_pt_dist(a,b,p):
    ab=b-a; t=np.clip(np.dot(p-a,ab)/np.dot(ab,ab),0,1); return np.linalg.norm(a+t*ab-p)
res=[]
for seed in range(40):
    obs,info=env.reset(seed=seed, options={'object_count':1})
    L=layout(obs); o=np.array([L['objective0']['x'],L['objective0']['y']])
    pill={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('obstacle') and L[n]['half_z']>0.1}
    mound={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('obstacle') and L[n]['half_z']<0.1}
    for qn,q in pill.items():
        dqo=np.linalg.norm(q-o)
        if dqo<0.3 or dqo>1.3: continue
        u=(q-o)/dqo
        for Lb in [0.5,0.8]:
            P=q+u*Lb
            if abs(P[0])<0.45 or abs(P[0])>2.0 or abs(P[1])>2.0: continue
            if np.sign(P[0])!=np.sign(o[0]) and abs(o[0])>0.3: continue
            d=np.linalg.norm(P-o)
            if d>1.8: continue
            bad=False
            for n2,p2 in pill.items():
                if n2==qn: continue
                if seg_pt_dist(P,o,p2)<0.35 or np.linalg.norm(P-p2)<0.4: bad=True
            for n2,m2 in mound.items():
                if seg_pt_dist(P,o,m2)<0.45 and np.linalg.norm(m2-o)>0.4: bad=True
            # lateral sweep space must be clear
            nrm=np.array([-u[1],u[0]])
            for s in [-0.45,0.45]:
                Ps=P+nrm*s
                if abs(Ps[0])<0.45 or abs(Ps[1])>2.0 or abs(Ps[0])>2.0: bad=True
                if any(np.linalg.norm(Ps-p2)<0.4 for p2 in pill.values()): bad=True
            if bad: continue
            res.append((seed,qn,Lb,tuple(np.round(P,3)),round(float(d),3),round(float(dqo),3),tuple(np.round(o,3))))
env.close()
print("cands",len(res))
for c in res[:15]: print(c)
