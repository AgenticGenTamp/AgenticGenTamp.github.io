import subprocess, sys, numpy as np
def load(path):
    out = subprocess.run(['convert', path, '-depth','8','rgb:-'], capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(360,640,3).astype(int)
base=load('/sandbox/mcp_renders/state_custom_policy_seed0_0000.png')
for f in ['0000','0003','0006','0009','0011']:
    p='/sandbox/mcp_renders/state_custom_policy_seed0_%s.png'%f
    img=load(p)
    d=np.abs(img-base).sum(2)
    m=d>25
    ys,xs=np.nonzero(m)
    if len(ys)==0:
        print(f,'no diff'); continue
    print(f,'ndiff=%d  x[%d-%d] y[%d-%d]'%(len(ys),xs.min(),xs.max(),ys.min(),ys.max()))
