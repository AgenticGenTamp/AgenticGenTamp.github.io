import numpy as np
from env_client import make_env
from pngtool import readpng
from collections import deque
env=make_env(); obs,_=env.reset(seed=0)
base=obs.copy(); base[125]=3.0; base[126]=-2.0   # move robot base far away
base[147:163]=0.0                                 # try to hide wiper
A=readpng(env.render_state(state=base.tolist(), label="di2_base")).astype(int)
darkA=(A.max(2)<70); H,W=darkA.shape
lab=-np.ones((H,W),int); blobs={}; n=0
for y in range(H):
  for x in range(W):
    if darkA[y,x] and lab[y,x]<0:
      q=deque([(y,x)]); lab[y,x]=n; pts=[]
      while q:
        cy,cx=q.popleft(); pts.append((cy,cx))
        for dy in(-1,0,1):
          for dx in(-1,0,1):
            ny,nx=cy+dy,cx+dx
            if 0<=ny<H and 0<=nx<W and darkA[ny,nx] and lab[ny,nx]<0:
              lab[ny,nx]=n; q.append((ny,nx))
      blobs[n]=np.array(pts); n+=1
def desc(p):
    cen=p.mean(0); pr=p-cen; cov=pr.T@pr/len(p); w,v=np.linalg.eigh(cov)
    d=v[:,-1]; t=pr@d; e1=cen+d*t.min(); e2=cen+d*t.max()
    return cen,(t.max()-t.min()),e1,e2,2*np.sqrt(max(w[0],1e-9))
print("all dark blobs in clean base:")
for k,p in blobs.items():
    if len(p)>=10:
        c,L,e1,e2,wid=desc(p)
        print(f'  blob{k} n={len(p)} c=({c[1]:.1f},{c[0]:.1f}) len={L:.1f} wid={wid:.1f} end=({e1[1]:.1f},{e1[0]:.1f})-({e2[1]:.1f},{e2[0]:.1f})')
print()
for j in range(6):
    o=base.copy(); o[103+j]=0.02
    B=readpng(env.render_state(state=o.tolist(), label="di2_%d"%j)).astype(int)
    d=(np.abs(B-A).max(2)>25)
    ids={}
    for k in np.unique(lab[d]):
        if k>=0: ids[k]=int((lab[d]==k).sum())
    print(f'--- j={j}')
    for k,cnt in sorted(ids.items(), key=lambda t:-t[1])[:3]:
        c,L,e1,e2,wid=desc(blobs[k])
        print(f'   blob{k} n={len(blobs[k])} moved={cnt} center=({c[1]:.2f},{c[0]:.2f}) len={L:.1f} wid={wid:.1f} end=({e1[1]:.1f},{e1[0]:.1f})-({e2[1]:.1f},{e2[0]:.1f})')
env.close()
