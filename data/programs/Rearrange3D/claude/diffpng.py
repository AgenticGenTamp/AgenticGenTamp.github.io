import numpy as np, sys
from png_tool import read_png, write_png
D='/sandbox/mcp_renders/state_custom_policy_seed0_%04d.png'
def L(i): return read_png(D%i).astype(float)
a=L(35); b=L(79); c=L(38)
print("shape",a.shape)
def bbox(x,y,tag):
    d=np.abs(x-y).sum(axis=2)
    m=d>20
    print(tag,"npix>20:",m.sum(),"maxdiff",d.max())
    if m.sum():
        ys,xs=np.nonzero(m)
        print("  bbox y",ys.min(),ys.max(),"x",xs.min(),xs.max())
        return ys.min(),ys.max(),xs.min(),xs.max()
    return None
print("CONTROL (both grip=0, t35 vs t38):"); bbox(a,c,"ctrl")
print("TEST (grip0 t35 vs grip1 t79):"); bb=bbox(a,b,"test")
if bb:
    y0,y1,x0,x1=bb
    pad=25
    y0=max(0,y0-pad);x0=max(0,x0-pad);y1=min(a.shape[0]-1,y1+pad);x1=min(a.shape[1]-1,x1+pad)
    def crop(img,name):
        sub=img[y0:y1+1,x0:x1+1].astype(np.uint8)
        z=np.repeat(np.repeat(sub,6,axis=0),6,axis=1)
        write_png(name,z); print("wrote",name,z.shape)
    crop(a,'/sandbox/mcp_renders/zz_grip0.png')
    crop(b,'/sandbox/mcp_renders/zz_grip1.png')
