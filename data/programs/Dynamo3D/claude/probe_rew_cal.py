import numpy as np, glob, json
from PIL import Image
fs=sorted(glob.glob("mcp_renders/state_custom_policy_seed1_*_1.png"))
ims=np.stack([np.asarray(Image.open(f).convert("RGB")).astype(float) for f in fs])
med=np.median(ims,axis=0)
cents=[]
for i,f in enumerate(fs):
    d=np.abs(ims[i]-med).sum(axis=2)
    m=d>40
    ys,xs=np.nonzero(m)
    cents.append((round(xs.mean(),1),round(ys.mean(),1),len(xs)))
    print(i,cents[-1])
log=[json.loads(l) for l in open("probe_rew_ap/log.jsonl")]
print("log len",len(log))
for k in range(0,90,9):
    print("step",k,round(log[k]["bx"],3),round(log[k]["by"],3))
