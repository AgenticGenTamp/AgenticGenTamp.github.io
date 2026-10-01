import numpy as np, json
M=np.load('cam_M.npy'); d=json.load(open('probe_drawer_tri.json'))
names={'0':'s0c0','1':'s0c1','2':'s0c2','3':'s1c0','4':'s1c1','5':'s1c2'}
def rows(uv, off=np.zeros(3)):
    u,v=uv
    r1=M[0]-u*M[2]; r2=M[1]-v*M[2]
    # (r[:3]).(P+off) + r[3] = 0
    return [(r1[:3], r1[3]+r1[:3]@off), (r2[:3], r2[3]+r2[:3]@off)]
def plane_bp(uv, x):
    u,v=uv
    r1=M[0]-u*M[2]; r2=M[1]-v*M[2]
    A=np.array([r1[1:3], r2[1:3]]); b=-np.array([r1[3]+r1[0]*x, r2[3]+r2[0]*x])
    return np.linalg.solve(A,b)
D=0.30
print(f"{'drw':6s} {'triangulated x,y,z':>26s} {'reproj_res_px':>13s}   plane x=0.87 (y,z)   bar y-extent(len m) @x=0.87")
out={}
for k in sorted(d, key=int):
    e=d[k]
    R=rows(e['closed_c'])+rows(e['open_c'], off=np.array([D,0,0]))
    A=np.array([r[0] for r in R]); b=-np.array([r[1] for r in R])
    P,*_=np.linalg.lstsq(A,b,rcond=None)
    resid=A@P-b
    # reprojection error in px
    def pr(X):
        h=M@np.r_[X,1]; return np.array([h[0]/h[2],h[1]/h[2]])
    ec=np.linalg.norm(pr(P)-np.array(e['closed_c'])); eo=np.linalg.norm(pr(P+np.array([D,0,0]))-np.array(e['open_c']))
    yz87=plane_bp(e['closed_c'],0.87)
    y1=plane_bp(e['closed_e1'],0.87); y2=plane_bp(e['closed_e2'],0.87)
    print(f"{names[k]:6s} ({P[0]:6.3f},{P[1]:7.3f},{P[2]:6.3f})   c={ec:.2f} o={eo:.2f}   ({yz87[0]:7.3f},{yz87[1]:6.3f})   y {min(y1[0],y2[0]):.3f}..{max(y1[0],y2[0]):.3f} ({abs(y1[0]-y2[0]):.3f}m) dz={abs(y1[1]-y2[1]):.3f}")
    out[names[k]]=dict(P=P.tolist(), yz87=yz87.tolist(), e1=y1.tolist(), e2=y2.tolist())
json.dump(out, open('probe_drawer_world.json','w'), indent=1)
# sensitivity: how much does (y,z) move if we assume x=0.85 or 0.90
print()
for k in sorted(d,key=int):
    a=plane_bp(d[k]['closed_c'],0.85); b=plane_bp(d[k]['closed_c'],0.90)
    print(f"{names[k]:6s} x=0.85 -> ({a[0]:7.3f},{a[1]:6.3f})   x=0.90 -> ({b[0]:7.3f},{b[1]:6.3f})")
