import subprocess, numpy as np, glob, os
from collections import deque
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
for f in files:
    im=load(f);R,G,B=im[...,0],im[...,1],im[...,2]
    strict=((R<=32)&(np.abs(R-G)<=2)&(np.abs(G-B)<=2)&(np.abs(R-B)<=2))
    m=np.zeros_like(strict); m[20:155,235:330]=strict[20:155,235:330]
    # largest CC
    lab=np.zeros(m.shape,bool); best=None
    ys,xs=np.nonzero(m)
    for y0,x0 in zip(ys,xs):
        if lab[y0,x0]: continue
        q=deque([(y0,x0)]); lab[y0,x0]=True; pts=[]
        while q:
            y,x=q.popleft(); pts.append((y,x))
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 20<=ny<155 and 235<=nx<330 and m[ny,nx] and not lab[ny,nx]:
                        lab[ny,nx]=True; q.append((ny,nx))
        if best is None or len(pts)>len(best): best=pts
    a=np.array(best); ys2,xs2=a[:,0],a[:,1]; y1=ys2.max()
    lines=[]
    for yy in range(y1-3,y1+1):
        xr=np.sort(xs2[ys2==yy]); runs=[]
        if len(xr):
            s=xr[0];p=xr[0]
            for v in xr[1:]:
                if v>p+1: runs.append((int(s),int(p))); s=v
                p=v
            runs.append((int(s),int(p)))
        lines.append(f"y{yy}:{runs}")
    print(f"{os.path.basename(f)[-10:-6]} n{len(a):4d} bbox x{xs2.min():3d}-{xs2.max():3d} y{ys2.min():3d}-{y1:3d} centroid({xs2.mean():5.1f},{ys2.mean():5.1f}) | "+" ".join(lines))
