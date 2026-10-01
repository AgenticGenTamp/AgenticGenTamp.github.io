import numpy as np
from pngtool import readpng
a = readpng('mcp_renders/state_seed0_seed0_reset.png').astype(int)
m = (a.max(2) < 70)
print('dark px', m.sum())
lab = -np.ones(m.shape, int); nid=0
from collections import deque
H,W = m.shape
for y in range(H):
    for x in range(W):
        if m[y,x] and lab[y,x]<0:
            q=deque([(y,x)]); lab[y,x]=nid; pts=[]
            while q:
                cy,cx=q.popleft(); pts.append((cy,cx))
                for dy in(-1,0,1):
                    for dx in(-1,0,1):
                        ny,nx=cy+dy,cx+dx
                        if 0<=ny<H and 0<=nx<W and m[ny,nx] and lab[ny,nx]<0:
                            lab[ny,nx]=nid; q.append((ny,nx))
            pts=np.array(pts)
            if len(pts)>=8:
                ys,xs=pts[:,0],pts[:,1]
                c=pts.mean(0)
                p=pts-c; cov=p.T@p/len(p); w,v=np.linalg.eigh(cov)
                d=v[:,-1]; t=p@d
                e1=c+d*t.min(); e2=c+d*t.max()
                print(f'blob n={len(pts):4d} center uv=({c[1]:6.1f},{c[0]:6.1f}) bbox u[{xs.min()},{xs.max()}] v[{ys.min()},{ys.max()}] '
                      f'len={t.max()-t.min():5.1f} wid={2*np.sqrt(max(w[0],1e-9)):4.1f} '
                      f'end1=({e1[1]:6.1f},{e1[0]:6.1f}) end2=({e2[1]:6.1f},{e2[0]:6.1f})')
            nid+=1
