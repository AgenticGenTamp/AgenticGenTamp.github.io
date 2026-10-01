import numpy as np, json
from env_client import make_env
from pngtool import readpng
from collections import deque
M=np.load('cam_M.npy')
def proj(P):
    h=M@np.r_[np.asarray(P,float),1.0]; return h[0]/h[2], h[1]/h[2]
def blobs_of(img, minn=8):
    dk=(img.max(2)<70); H,W=dk.shape; lab=-np.ones((H,W),int); out=[]; n=0
    for y in range(H):
      for x in range(W):
        if dk[y,x] and lab[y,x]<0:
          q=deque([(y,x)]); lab[y,x]=n; pts=[]
          while q:
            cy,cx=q.popleft(); pts.append((cy,cx))
            for dy in(-1,0,1):
              for dx in(-1,0,1):
                ny,nx=cy+dy,cx+dx
                if 0<=ny<H and 0<=nx<W and dk[ny,nx] and lab[ny,nx]<0:
                  lab[ny,nx]=n; q.append((ny,nx))
          if len(pts)>=minn: out.append(np.array(pts))
          n+=1
    return out
def desc(p):
    c=p.mean(0); pr=p-c; cov=pr.T@pr/len(p); w,v=np.linalg.eigh(cov)
    d=v[:,-1]; t=pr@d
    return np.array([c[1],c[0]]), np.array([(c+d*t.min())[1],(c+d*t.min())[0]]), np.array([(c+d*t.max())[1],(c+d*t.max())[0]])

env=make_env(); obs,_=env.reset(seed=0)
base=obs.copy(); base[125]=3.0; base[126]=-2.0; base[147:163]=0.0
Aimg=readpng(env.render_state(state=base.tolist(), label="tri_base")).astype(int)
B0=blobs_of(Aimg)
closed={ }
names={0:'s0c0',1:'s0c1',2:'s0c2',3:'s1c0',4:'s1c1',5:'s1c2'}
ref={0:(416.80,310.52),1:(366.56,362.69),2:(305.95,425.71),3:(419.18,289.82),4:(368.03,341.35),5:(305.82,404.00)}
for j,r in ref.items():
    best=min(B0, key=lambda p: np.hypot(*(desc(p)[0]-np.array(r))))
    closed[j]=desc(best)
D=0.30
res={}
for j in range(6):
    o=base.copy(); o[103+j]=D
    Bimg=readpng(env.render_state(state=o.tolist(), label="tri_%d"%j)).astype(int)
    ch=(np.abs(Bimg-Aimg).max(2)>25)
    cand=[p for p in blobs_of(Bimg) if ch[p[:,0],p[:,1]].mean()>0.7]
    if not cand: print(j,'no moved blob'); continue
    # pick candidate with similar pixel count
    n0=len(min(B0,key=lambda p: np.hypot(*(desc(p)[0]-np.array(ref[j])))))
    best=min(cand,key=lambda p: abs(len(p)-n0)+0.02*np.hypot(*(desc(p)[0]-closed[j][0])))
    op=desc(best)
    res[j]=(closed[j],op)
    print(f'j={j} {names[j]} closed_c={closed[j][0].round(2)} open_c={op[0].round(2)} shift={(op[0]-closed[j][0]).round(2)}')
np.save('probe_drawer_tri.npy', np.array([[*res[j][0][0],*res[j][1][0]] for j in sorted(res)]))
json.dump({str(j):dict(closed_c=res[j][0][0].tolist(),closed_e1=res[j][0][1].tolist(),closed_e2=res[j][0][2].tolist(),
                       open_c=res[j][1][0].tolist()) for j in res}, open('probe_drawer_tri.json','w'), indent=1)
env.close()
