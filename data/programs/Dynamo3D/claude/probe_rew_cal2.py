import numpy as np, json
from PIL import Image
fs=["mcp_renders/state_custom_policy_seed1_%04d_2.png"%i for i in range(10)]+ \
   ["mcp_renders/state_custom_policy_seed1_%04d_1.png"%i for i in range(10,20)]
log=[json.loads(l) for l in open("probe_rew_ap2/log.jsonl")]
N=len(log); print("steps logged",N)
def blobs(f):
    im=np.asarray(Image.open(f).convert("RGB")).astype(int)
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    green=(G>R+25)&(G>B+25)
    tan=(R>140)&(R<235)&(G>110)&(G<205)&(B<165)&(R>B+25)&(abs(R-G)<45)
    def cen(m):
        ys,xs=np.nonzero(m)
        return None if len(xs)==0 else (xs.mean(),ys.mean(),len(xs),xs.min(),xs.max(),ys.min(),ys.max())
    return cen(green),cen(tan)
step=max(1,N//20)
for i,f in enumerate(fs):
    g,t=blobs(f)
    k=min(i*step,N-1)
    print(i,"step~",k,"chair",round(log[k]["cx"],3),round(log[k]["cy"],3),
          "| tan",None if t is None else tuple(round(v,1) for v in t),
          "| green",None if g is None else tuple(round(v,1) for v in g))
