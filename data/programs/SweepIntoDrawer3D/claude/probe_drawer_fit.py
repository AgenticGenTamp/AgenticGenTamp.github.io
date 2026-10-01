import numpy as np, json
from scipy.optimize import least_squares
M=np.load('cam_M.npy'); d=json.load(open('probe_drawer_tri.json'))
names={'0':'s0c0','1':'s0c1','2':'s0c2','3':'s1c0','4':'s1c1','5':'s1c2'}
row={'0':0,'1':0,'2':0,'3':1,'4':1,'5':1}; col={'0':0,'1':1,'2':2,'3':0,'4':1,'5':2}
D=0.30
obs=[]
for k in sorted(d,key=int):
    obs.append((row[k],col[k],np.array(d[k]['closed_c']),np.array(d[k]['open_c'])))
def pr(P):
    h=M@np.r_[P,1.0]; return np.array([h[0]/h[2],h[1]/h[2]])
def resid(p, dirfree=True):
    X=p[0]; zl,zh=p[1],p[2]; ys=p[3:6]
    if dirfree: dvec=np.array([p[6],p[7],p[8]])
    else: dvec=np.array([1.0,0,0])
    r=[]
    for rw,cl,c,o in obs:
        P=np.array([X, ys[cl], zl if rw==0 else zh])
        r+= list(pr(P)-c); r+= list(pr(P+D*dvec)-o)
    return np.array(r)
p0=np.array([0.92,0.07,0.33,0.70,0.02,-0.63,1.0,0.0,0.0])
s1=least_squares(lambda p: resid(p,False), p0[:6])
print('fixed +x dir: X=%.4f zlow=%.4f zhigh=%.4f y=[%.4f %.4f %.4f] rms=%.3f px max=%.3f'%(
  s1.x[0],s1.x[1],s1.x[2],s1.x[3],s1.x[4],s1.x[5], np.sqrt((s1.fun**2).mean()*2), np.abs(s1.fun).max()))
s2=least_squares(resid, p0)
print('free dir    : X=%.4f zlow=%.4f zhigh=%.4f y=[%.4f %.4f %.4f] dir=[%.3f %.3f %.3f] rms=%.3f max=%.3f'%(
  s2.x[0],s2.x[1],s2.x[2],s2.x[3],s2.x[4],s2.x[5],*(s2.x[6:9]/np.linalg.norm(s2.x[6:9])), np.sqrt((s2.fun**2).mean()*2), np.abs(s2.fun).max()))
# per-observation residuals for fixed-dir fit
print('per handle reproj err (closed,open) px:')
f=s1.fun.reshape(6,4)
for i,k in enumerate(sorted(d,key=int)):
    print('  %s  %.2f  %.2f'%(names[k], np.hypot(*f[i,:2]), np.hypot(*f[i,2:])))
# uncertainty: jitter pixels 0.5px
rng=np.random.default_rng(0); P=[]
base=[(r,c,cc.copy(),oo.copy()) for r,c,cc,oo in obs]
for it in range(200):
    for i,(r,c,cc,oo) in enumerate(base):
        obs[i]=(r,c,cc+rng.normal(0,0.5,2), oo+rng.normal(0,0.5,2))
    P.append(least_squares(lambda p: resid(p,False), s1.x).x)
obs[:] = base
P=np.array(P); print('1-sigma from 0.5px pixel noise:', np.round(P.std(0),4))
X,zl,zh,y0,y1,y2=s1.x
json.dump(dict(X=X,z_low=zl,z_high=zh,y=[y0,y1,y2],sigma=P.std(0).tolist()), open('probe_drawer_fit.json','w'), indent=1)
# endpoints in world using fitted plane
def plane_bp(uv,x):
    u,v=uv; r1=M[0]-u*M[2]; r2=M[1]-v*M[2]
    A=np.array([r1[1:3],r2[1:3]]); b=-np.array([r1[3]+r1[0]*x, r2[3]+r2[0]*x])
    return np.linalg.solve(A,b)
print('\nbar endpoints back-projected on plane x=%.3f:'%X)
for k in sorted(d,key=int):
    a=plane_bp(d[k]['closed_e1'],X); b=plane_bp(d[k]['closed_e2'],X); c=plane_bp(d[k]['closed_c'],X)
    print('  %s center(y,z)=(%.3f,%.3f)  y %.3f..%.3f  len=%.3f m  z %.3f..%.3f'%(
      names[k],c[0],c[1],min(a[0],b[0]),max(a[0],b[0]),abs(a[0]-b[0]),min(a[1],b[1]),max(a[1],b[1])))
