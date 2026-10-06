import numpy as np, itertools, time
from kin import Kin, rot_pitch
k=Kin()
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
CONT=np.array([1,0,1,0,1,0,1],bool)
LIM=np.array([9,2.2,9,2.5,9,2.05,9])
def qdiff(a,b):
    dd=a-b; dd[CONT]=(dd[CONT]+np.pi)%(2*np.pi)-np.pi; return dd
ZPICK=0.02; ZPLACE=0.578
def pick_ok(q):
    pts=k.arm_points(q)
    if min(p[:,2].min() for p in pts)<0.003: return False
    for p in pts:
        if np.any((p[:,0]<0.29)&(p[:,2]<0.39)): return False
    return True
def place_ok(q, rb):
    front=rb-(1.42-1.373)-0.01
    for p in k.arm_points(q):
        if p[:,0].max()>front and (p[:,2].max()>0.79 or p[:,2].min()<0.556): return False
    return True
def is_sym(R):
    return np.allclose(np.abs(R).round(6).sum(0),1) and np.allclose(np.abs(np.round(R,6)).max(0),1)
res=[]
seeds=[q0, np.array([0,0.5,3.142,-2.0,0,-1.0,1.571]), np.array([0,1.0,3.142,-1.5,0,0.5,1.571])]
t=time.time()
for tha in np.linspace(-0.5,0.1,13):
  for sa in [1,-1]:
    Ra=rot_pitch(tha,sa)
    for ra in np.linspace(0.26,0.40,8):
      qa=None
      for sd in seeds:
        q,ok=k.ik(np.array([ra,0,ZPICK]),Ra,sd)
        if ok and np.all(np.abs(q)<LIM) and pick_ok(q): qa=q;break
      if qa is None: continue
      for thb in [tha+np.pi/2]:
        for sb in [1,-1]:
          Rb=rot_pitch(thb,sb)
          if not is_sym(Rb@Ra.T): continue
          for rb in np.linspace(0.8,1.05,11):
            qb,ok=k.ik(np.array([rb,0,ZPLACE]),Rb,qa,null_target=qa)
            if not ok or not np.all(np.abs(qb)<LIM) or not place_ok(qb,rb): continue
            c=np.abs(qdiff(qb,qa)).max()
            res.append((c,tha,sa,ra,thb,sb,rb,qa,qb))
print(time.time()-t, len(res))
res.sort(key=lambda r:r[0])
for r in res[:15]:
    print(round(r[0],3),'tha',round(r[1],2),r[2],'ra',round(r[3],2),'thb',round(r[4],2),r[5],'rb',round(r[6],2), np.round(qdiff(r[8],r[7]),2))
import pickle; pickle.dump(res[:50],open('grid_res3.pkl','wb'))
