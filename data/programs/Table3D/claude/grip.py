import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
im=load(files[-1])
sub=im[80:130,240:320]
vals,counts=np.unique(sub.reshape(-1,3),axis=0,return_counts=True)
o=np.argsort(-counts)[:15]
for i in o: print(vals[i],counts[i])
