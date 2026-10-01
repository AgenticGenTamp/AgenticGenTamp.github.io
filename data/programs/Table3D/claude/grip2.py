import subprocess, numpy as np, glob, os
from collections import deque
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
def comps(mask,minsize=20):
    lab=-np.ones(mask.shape,int); out=[]
    ys,xs=np.nonzero(mask)
    for y0,x0 in zip(ys,xs):
        if lab[y0,x0]!=-1: continue
        q=deque([(y0,x0)]); lab[y0,x0]=1; pts=[]
        while q:
            y,x=q.popleft(); pts.append((y,x))
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<H and 0<=nx<W and mask[ny,nx] and lab[ny,nx]==-1:
                        lab[ny,nx]=1; q.append((ny,nx))
        if len(pts)>=minsize:
            a=np.array(pts); out.append((len(pts),a))
    return sorted(out,key=lambda t:-t[0])
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
print("frame  npx  bbox_x0 x1  y0 y1   cx    cy   bottom_row_xspan   width_at_bottom")
for f in files:
    im=load(f);R,G,B=im[...,0],im[...,1],im[...,2]
    black=(R<=30)&(np.abs(R-G)<=2)&(np.abs(G-B)<=2)&(np.abs(R-B)<=2)
    cs=comps(black,30)
    n,a=cs[0]
    y0,y1,x0,x1=a[:,0].min(),a[:,0].max(),a[:,1].min(),a[:,1].max()
    br=a[a[:,0]>=y1-2]
    print(f"{os.path.basename(f)[-10:-6]}  {n:5d}  {x0:4d} {x1:4d}  {y0:4d} {y1:4d}  {a[:,1].mean():6.1f} {a[:,0].mean():6.1f}   x[{br[:,1].min()}-{br[:,1].max()}]  {br[:,1].max()-br[:,1].min()+1}  ncomp={len(cs)}")
