import numpy as np, glob, json
from PIL import Image
frames=sorted(glob.glob("mcp_renders/state_custom_policy_seed1_*.png"))
for f in frames:
    im=np.asarray(Image.open(f).convert("RGB")).astype(int)
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    green=(G>R+25)&(G>B+25)
    tan=(R>140)&(R<235)&(G>110)&(G<205)&(B<165)&(R>B+25)&(abs(R-G)<45)
    def cen(m):
        ys,xs=np.nonzero(m)
        return None if len(xs)==0 else (round(xs.mean(),1),round(ys.mean(),1),len(xs))
    print(f[-12:],"green",cen(green),"tan",cen(tan))
