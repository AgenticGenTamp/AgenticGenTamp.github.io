import numpy as np, json
from PIL import Image
log=[json.loads(l) for l in open("probe_rew_ap2/log.jsonl")]
print("steps",len(log))
def blobs(f):
    im=np.asarray(Image.open(f).convert("RGB")).astype(int)
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    green=(G>R+25)&(G>B+25)
    tan=(R>140)&(R<235)&(G>110)&(G<205)&(B<165)&(R>B+25)&(abs(R-G)<45)
    def cen(m):
        ys,xs=np.nonzero(m)
        return None if len(xs)==0 else (round(xs.mean(),1),round(ys.mean(),1),len(xs))
    return cen(green),cen(tan)
for i in range(0,65,4):
    g,t=blobs("mcp_renders/state_custom_policy_seed11_%04d.png"%i)
    k=min(i,len(log)-1)
    print(i,"chair",round(log[k]["cx"],3),round(log[k]["cy"],3),"base",round(log[k]["bx"],2),round(log[k]["by"],2),"| tan",t,"| green",g)
