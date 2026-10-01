import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
base=load(files[0])
for f in files:
    im=load(f)
    d=np.abs(im-base).sum(2)
    m=d>25
    n=m.sum()
    if n:
        ys,xs=np.nonzero(m)
        print(os.path.basename(f),'nchanged',n,'bbox x',xs.min(),xs.max(),'y',ys.min(),ys.max())
    else:
        print(os.path.basename(f),'identical to frame0')
