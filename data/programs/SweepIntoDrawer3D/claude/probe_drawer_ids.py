import numpy as np
from env_client import make_env
from pngtool import readpng
from collections import deque
env=make_env(); obs,_=env.reset(seed=0)
A=readpng(env.render_state(state=obs.tolist(), label="di_base")).astype(int)
darkA=(A.max(2)<70)
H,W=darkA.shape
# label darkA
lab=-np.ones((H,W),int); blobs={}
n=0
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
for j in range(6):
    o=obs.copy(); o[103+j]=0.02
    B=readpng(env.render_state(state=o.tolist(), label="di_%d"%j)).astype(int)
    d=(np.abs(B-A).max(2)>25)
    ids={}
    for k in np.unique(lab[d]):
        if k>=0: ids[k]=int((lab[d]==k).sum())
    print(f'--- j={j} (obs103+{j})')
    for k,c in sorted(ids.items(), key=lambda t:-t[1])[:4]:
        p=blobs[k]; cen=p.mean(0); pr=p-cen; cov=pr.T@pr/len(p); w,v=np.linalg.eigh(cov)
        dvec=v[:,-1]; t=pr@dvec; e1=cen+dvec*t.min(); e2=cen+dvec*t.max()
        print(f'   blob{k} nblob={len(p)} moved={c} center=({cen[1]:.1f},{cen[0]:.1f}) len={t.max()-t.min():.1f} '
              f'end=({e1[1]:.1f},{e1[0]:.1f})-({e2[1]:.1f},{e2[0]:.1f})')
env.close()
