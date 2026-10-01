import numpy as np
from probe_vis_lib import *

def seg_pt_dist(a,b,p):
    ab=b-a; t=np.clip(np.dot(p-a,ab)/np.dot(ab,ab),0,1); return np.linalg.norm(a+t*ab-p)

def reachable(P):
    return abs(P[0])>0.30 and abs(P[0])<2.10 and abs(P[1])<2.10

env=make_env()
cands=[]; mcands=[]
for seed in range(25):
    obs,info=env.reset(seed=seed)
    L=layout(obs)
    objs={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('objective')}
    pill={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('obstacle') and L[n]['half_z']>0.1}
    mound={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('obstacle') and L[n]['half_z']<0.1}
    for on,o in objs.items():
        for qn,q in pill.items():
            dqo=np.linalg.norm(q-o)
            if dqo<0.3 or dqo>1.5: continue
            u=(q-o)/dqo
            for s in [0.4,0.6,0.8]:
                P=q+u*s
                if not reachable(P): continue
                if np.sign(P[0])!=np.sign(o[0]): continue
                d=np.linalg.norm(P-o)
                if d>1.85: continue
                # other objectives out of range
                if any(np.linalg.norm(P-o2)<2.06 for n2,o2 in objs.items() if n2!=on): continue
                # only q near segment; other pillars far; mounds far
                bad=False
                for n2,p2 in list(pill.items()):
                    if n2==qn and True: continue
                    if seg_pt_dist(P,o,p2)<0.35: bad=True
                for n2,p2 in mound.items():
                    if seg_pt_dist(P,o,p2)<0.45: bad=True
                # pillar not too close to rover for collisions
                if np.linalg.norm(P-q)<0.35: bad=True
                if bad: continue
                cands.append((seed,on,qn,round(float(s),2),tuple(np.round(P,3)),round(float(d),3),round(float(dqo),3)))
    # mound occlusion candidates: segment passes through a mound that is NOT under target objective
    for on,o in objs.items():
        for mn,m in mound.items():
            if np.linalg.norm(m-o)<0.4: continue
            # stand on far side of mound along line
            u=(m-o)/np.linalg.norm(m-o)
            for s in [0.5,0.8]:
                P=m+u*s
                if not reachable(P): continue
                if np.sign(P[0])!=np.sign(o[0]): continue
                d=np.linalg.norm(P-o)
                if d>1.85: continue
                if any(np.linalg.norm(P-o2)<2.06 for n2,o2 in objs.items() if n2!=on): continue
                if any(seg_pt_dist(P,o,p2)<0.35 for p2 in pill.values()): continue
                if any(seg_pt_dist(P,o,m2)<0.35 for n2,m2 in mound.items() if n2!=mn): continue
                mcands.append((seed,on,mn,round(float(s),2),tuple(np.round(P,3)),round(float(d),3)))
env.close()
print("PILLAR cands",len(cands))
for c in cands[:12]: print(c)
print("MOUND cands",len(mcands))
for c in mcands[:8]: print(c)
