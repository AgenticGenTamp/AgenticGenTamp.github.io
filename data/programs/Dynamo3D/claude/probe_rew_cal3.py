import numpy as np, json
from PIL import Image
log=[json.loads(l) for l in open("probe_rew_ap2/log.jsonl")]
N=len(log); print("steps",N,"first",log[0],"last",log[-1])
fs=["mcp_renders/state_custom_policy_seed11_%04d.png"%i for i in range(20)]
def blobs(f):
    im=np.asarray(Image.open(f).convert("RGB")).astype(int)
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    green=(G>R+25)&(G>B+25)
    tan=(R>140)&(R<235)&(G>110)&(G<205)&(B<165)&(R>B+25)&(abs(R-G)<45)
    def cen(m):
        ys,xs=np.nonzero(m)
        return None if len(xs)==0 else (round(xs.mean(),1),round(ys.mean(),1),len(xs))
    return cen(green),cen(tan)
for i,f in enumerate(fs):
    g,t=blobs(f)
    k=min(int(round(i*(N-1)/19.0)),N-1)
    print(i,"step",k,"chair",round(log[k]["cx"],3),round(log[k]["cy"],3),"base",round(log[k]["bx"],2),round(log[k]["by"],2),"| tan",t,"| green",g)
