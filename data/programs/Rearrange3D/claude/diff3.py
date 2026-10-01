import numpy as np
from png_tool import read_png, write_png
D='/sandbox/mcp_renders/state_custom_policy_seed0_%04d.png'
y0,y1,x0,x1=248,335,388,436
def crop(i):
    im=read_png(D%i)[y0:y1,x0:x1].astype(np.uint8)
    return np.repeat(np.repeat(im,10,axis=0),10,axis=1)
A=crop(35); B=crop(79)
sep=np.full((A.shape[0],20,3),255,np.uint8)
write_png('/sandbox/mcp_renders/zz_sbs.png', np.concatenate([A,sep,B],axis=1))
print("left=grip0.0 right=grip1.0", A.shape)
