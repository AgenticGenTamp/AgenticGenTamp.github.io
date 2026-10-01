import subprocess, numpy as np, glob, os
W,H=640,360
def load(p):
    raw=subprocess.run(['convert',p,'-depth','8','rgb:-'],capture_output=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(H,W,3).astype(int)
files=sorted(glob.glob('mcp_renders/state_custom_policy_seed0_000?_1.png'))
print("frm  npx   x0  x1   y0  y1    cx     cy   bottomrow_x   left_tip  right_tip")
for f in files:
    im=load(f);R,G,B=im[...,0],im[...,1],im[...,2]
    black=(R<=32)&(np.abs(R-G)<=2)&(np.abs(G-B)<=2)&(np.abs(R-B)<=2)
    m=np.zeros_like(black); m[40:210,200:460]=black[40:210,200:460]
    ys,xs=np.nonzero(m)
    y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    br=xs[ys>=y1-1]
    # bright fingertip pixels inside/near gripper bbox
    tipm=(R>200)&(np.abs(R-G)<8)&(np.abs(G-B)<8)
    box=np.zeros_like(tipm); box[y0:y1+3,x0-3:x1+4]=tipm[y0:y1+3,x0-3:x1+4]
    tys,txs=np.nonzero(box)
    s=f"{os.path.basename(f)[-10:-6]} {len(xs):5d} {x0:4d}{x1:4d} {y0:4d}{y1:4d} {xs.mean():6.1f}{ys.mean():7.1f}  x[{br.min()}-{br.max()}]"
    if len(txs): s+=f"   tips n={len(txs)} x[{txs.min()}-{txs.max()}] y[{tys.min()}-{tys.max()}]"
    print(s)
