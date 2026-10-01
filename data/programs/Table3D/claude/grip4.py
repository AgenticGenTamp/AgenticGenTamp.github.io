import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
ims=[load(f) for f in files]
def bm(im):
    R,G,B=im[...,0],im[...,1],im[...,2]
    return (R<=32)&(np.abs(R-G)<=2)&(np.abs(G-B)<=2)&(np.abs(R-B)<=2)
masks=[bm(i) for i in ims]
static=np.all(np.stack(masks),0)
print('static black px',static.sum())
print("frm  npx    x0  x1   y0  y1    cx     cy    bottom2rows_x   width  tips")
for f,m,im in zip(files,masks,ims):
    g=m&~static
    g[:, :200]=False; g[:,470:]=False; g[210:,:]=False
    ys,xs=np.nonzero(g)
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    br=xs[ys>=y1-1]
    R,G,B=im[...,0],im[...,1],im[...,2]
    tipm=(R>200)&(np.abs(R-G)<8)&(np.abs(G-B)<8)
    box=np.zeros_like(tipm); box[max(0,y0):y1+2, x0-2:x1+3]=tipm[max(0,y0):y1+2, x0-2:x1+3]
    tys,txs=np.nonzero(box)
    s=f"{os.path.basename(f)[-10:-6]} {len(xs):5d}  {x0:4d}{x1:4d} {y0:4d}{y1:4d} {xs.mean():6.1f}{ys.mean():7.1f}   x[{br.min()}-{br.max()}] {br.max()-br.min()+1:3d}"
    if len(txs): s+=f"  n={len(txs)} x[{txs.min()}-{txs.max()}] y[{tys.min()}-{tys.max()}]"
    print(s)
