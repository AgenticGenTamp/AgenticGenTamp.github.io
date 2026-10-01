import numpy as np
exec(open('tri.py').read().split("dirs=")[0])
for j in range(6):
    ma=dm('mcp_renders/state_custom_cl_d%da.png'%j); mb=dm('mcp_renders/state_custom_cl_d%db.png'%j)
    pts=[]
    for a in np.linspace(-np.pi/2,np.pi/2,49):
        d=(np.cos(a),np.sin(a)); P=[]
        for m in (ma,mb):
            ys,xs=np.nonzero(m); s=d[0]*xs+d[1]*ys; k=s.max(); sel=s>=k-0.5
            P.append((xs[sel].mean(), ys[sel].mean()))
        if abs(P[0][0]-P[1][0])<3 and abs(P[0][1]-P[1][1])<3: continue
        X,r=tri(P[0],P[1])
        if r>0.05: continue
        if not (1.15<X[0]<1.55): continue
        pts.append(X)
    pts=np.array(pts)
    print('idx',103+j,'n',len(pts))
    if len(pts):
        print('   x: %.3f..%.3f med %.3f | y: %.3f..%.3f | z: %.3f..%.3f'%(pts[:,0].min(),pts[:,0].max(),np.median(pts[:,0]),pts[:,1].min(),pts[:,1].max(),pts[:,2].min(),pts[:,2].max()))
        # print distinct clusters
        u=np.unique(np.round(pts,2),axis=0)
        print('   corners:',[tuple(v) for v in u[:12]])
