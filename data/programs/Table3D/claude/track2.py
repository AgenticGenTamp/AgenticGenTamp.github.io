import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
ims=[load(f) for f in files]
def gm(im):
    R,G,B=im[...,0],im[...,1],im[...,2]
    return (np.abs(R-G)<=6)&(np.abs(G-B)<=6)&(np.abs(R-B)<=6)&(R<=215)
ms=[gm(i) for i in ims]
static=np.all(np.stack(ms),0)
for f,m in zip(files,ms):
    g=m&~static
    mm=np.zeros_like(g); mm[70:150,240:320]=g[70:150,240:320]
    ys,xs=np.nonzero(mm)
    y1=ys.max()
    lines=[]
    for yy in range(y1-3,y1+1):
        xr=np.sort(xs[ys==yy]); runs=[]
        if len(xr):
            s=xr[0];p=xr[0]
            for v in xr[1:]:
                if v>p+1: runs.append((int(s),int(p))); s=v
                p=v
            runs.append((int(s),int(p)))
        lines.append(f"y{yy}:{runs}")
    print(f"{os.path.basename(f)[-10:-6]} bbox x{xs.min():3d}-{xs.max():3d} y{ys.min():3d}-{y1:3d} cx{xs.mean():6.1f} npx{len(xs):4d} | "+" ".join(lines))
