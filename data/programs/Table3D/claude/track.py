import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
print("frm | gripper bbox x0-x1 y0-y1 | cx | lowest y | x-runs on lowest 3 rows | finger gap")
for f in files:
    im=load(f);R,G,B=im[...,0],im[...,1],im[...,2]
    neutral=(np.abs(R-G)<=6)&(np.abs(G-B)<=6)&(np.abs(R-B)<=6)
    g=neutral&(R<=215)
    m=np.zeros_like(g); m[75:145,250:312]=g[75:145,250:312]
    ys,xs=np.nonzero(m)
    y1=ys.max()
    out=[]
    for yy in range(y1-2,y1+1):
        xr=np.sort(xs[ys==yy])
        runs=[];  
        if len(xr):
            s=xr[0];p=xr[0]
            for v in xr[1:]:
                if v>p+1: runs.append((s,p)); s=v
                p=v
            runs.append((s,p))
        out.append((yy,runs))
    # finger gap on lowest row with >=2 runs
    gap=''
    for yy,runs in reversed(out):
        if len(runs)>=2:
            gap=f"y{yy}: L={runs[0]} R={runs[-1]} gap={runs[-1][0]-runs[0][1]-1}px"
            break
    print(f"{os.path.basename(f)[-10:-6]} | x {xs.min():3d}-{xs.max():3d} y {ys.min():3d}-{y1:3d} | cx {xs.mean():5.1f} | ylow {y1:3d} | "+ " ".join(f"y{yy}{runs}" for yy,runs in out) + " | "+gap)
