import numpy as np
from pngtool import readpng
M=np.load('cam_M.npy')
ref=readpng('mcp_renders/state_custom_cl_ref.png').astype(int)
def dm(p):
    im=readpng(p).astype(int); return np.abs(im-ref).sum(2)>18
def tri(uv1,uv2,shift=0.20):
    A=[];b=[]
    for uv,off in [(uv1,0.0),(uv2,shift)]:
        u,v=uv
        # (m0 - u*m2).X = -(m0_3 - u*m2_3)  with X shifted in x
        r0=M[0,:3].copy(); r1=M[1,:3].copy(); r2=M[2,:3].copy()
        c0=M[0,3]+r0[0]*off; c1=M[1,3]+r1[0]*off; c2=M[2,3]+r2[0]*off
        A.append(r0-u*r2); b.append(-(c0-u*c2))
        A.append(r1-v*r2); b.append(-(c1-v*c2))
    A=np.array(A); b=np.array(b)
    X,*_=np.linalg.lstsq(A,b,rcond=None)
    res=A@X-b
    return X, np.abs(res).max()
dirs={'right':(1,0),'downright':(0.7,0.7),'upright':(0.7,-0.7),'down':(0,1),'up':(0,-1)}
for j in range(6):
    ma=dm('mcp_renders/state_custom_cl_d%da.png'%j); mb=dm('mcp_renders/state_custom_cl_d%db.png'%j)
    print('--- drawer idx',103+j)
    for nm,d in dirs.items():
        pts=[]
        for m in (ma,mb):
            ys,xs=np.nonzero(m)
            s=d[0]*xs+d[1]*ys
            k=s.max(); sel=s>=k-0.5
            pts.append((xs[sel].mean(), ys[sel].mean()))
        X,r=tri(pts[0],pts[1])
        print('  %-9s uv45=(%.0f,%.0f) uv65=(%.0f,%.0f) -> X=(%.3f,%.3f,%.3f) res=%.2f'%(nm,pts[0][0],pts[0][1],pts[1][0],pts[1][1],X[0],X[1],X[2],r))
