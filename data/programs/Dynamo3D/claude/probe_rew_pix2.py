import numpy as np
from PIL import Image
import glob
for f in sorted(glob.glob("mcp_renders/state_seed*.png")):
    im=np.asarray(Image.open(f).convert("RGB")).astype(int)
    R,G,B=im[:,:,0],im[:,:,1],im[:,:,2]
    green=(G>R+25)&(G>B+25)
    tan=(R>140)&(R<235)&(G>110)&(G<205)&(B<165)&(R>B+25)&(abs(R-G)<45)
    def cen(m):
        ys,xs=np.nonzero(m)
        if len(xs)==0: return None
        return (round(xs.mean(),1),round(ys.mean(),1),len(xs))
    print(f.split('/')[-1],"green",cen(green),"tan",cen(tan))
