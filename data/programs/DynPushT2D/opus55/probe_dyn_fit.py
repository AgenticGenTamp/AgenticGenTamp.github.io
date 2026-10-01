import numpy as np, glob
from scipy.optimize import least_squares
from probe_dyn_lib import rot, wrap
np.set_printoptions(precision=4,suppress=True)
D=np.concatenate([np.load(f) for f in sorted(glob.glob('probe_dyn_data_*.npy'))])
# cols: seed e k | bx by bt(3:6) | rx ry(6:8) | ux uy(8:10) | b2(10:13) | r2(13:15) | w L lv(15:18) | v ang
def closest(q,w,L,lv):
    best=None
    for x0,x1,y0,y1 in [(-L/2,L/2,-w,0),(-w/2,w/2,-w-lv,-w)]:
        c=np.array([np.clip(q[0],x0,x1),np.clip(q[1],y0,y1)]); d=np.linalg.norm(q-c)
        if best is None or d<best[0]: best=(d,c)
    d,c=best; return c,(q-c)/max(d,1e-9),d
rows=[]
moved=np.abs(D[:,10:13]-D[:,3:6]).max(1)>1e-8
for i in range(len(D)):
    r=D[i]
    if not moved[i] or i==0 or not moved[i-1] or D[i-1,1]!=r[1]: continue
    R=rot(r[5]).T
    q=R@(r[6:8]-r[3:5]); u=R@r[8:10]
    tw=np.r_[R@(r[10:12]-r[3:5]), wrap(r[12]-r[5])]
    p,n,d=closest(q,*r[15:18])
    rows.append(np.r_[p,n,u,tw,r[15:18],d,r[18],r[19],r[0]])
X=np.array(rows); print('steady contact steps',len(X))
P,N,U,TW,G,DD=X[:,0:2],X[:,2:4],X[:,4:6],X[:,6:9],X[:,9:12],X[:,12]
print('robot-surface distance at pre-step (gap+r): per speed')
for v in [0.01,0.02,0.03,0.0499]:
    m=np.isclose(X[:,13],v); print(' v',v,'dist-0.1 mean',(DD[m]-0.1).mean().round(4),'n',m.sum())
def centroid(w,L,lv):
    A1,A2=L*w,w*lv; cy=(A1*(-w/2)+A2*(-w-lv/2))/(A1+A2)
    I1=A1*((L**2+w**2)/12+(w/2)**2); I2=A2*((w**2+lv**2)/12+(w+lv/2)**2)
    return cy,(I1+I2)/(A1+A2)-cy**2
def predict(th,X,ret_mode=False):
    dcx,dcy,sc2,mu=th; out=[];modes=[]
    for x in X:
        p,n,u=x[0:2],x[2:4],x[4:6]; w,L,lv=x[9:12]
        cy0,c20=centroid(w,L,lv); c0=np.array([dcx,cy0+dcy]); c2=c20*sc2
        rr=p-c0; rp=np.array([-rr[1],rr[0]])
        M=c2*np.eye(2)+np.outer(rp,rp)
        f=np.linalg.solve(M,u); fn=-f@n; t=np.array([-n[1],n[0]]); ft=f@t
        mode=0
        if abs(ft)>mu*fn:
            # sliding: force on cone edge; sign of ft same as sticking solution
            f=-n+np.sign(ft)*mu*t; vp=M@f; lam=(u@n)/(vp@n); f=f*lam; mode=1
        V=c2*f; om=rr[0]*f[1]-rr[1]*f[0]
        Vo=V+om*np.array([c0[1],-c0[0]])  # om x (0-c0)
        out.append(np.r_[Vo,om]); modes.append(mode)
    return (np.array(out),np.array(modes)) if ret_mode else np.array(out)
def res(th): return ((predict(th,X)-X[:,6:9])/X[:,13:14]).ravel()
for th0 in [[0,0,1,0.5]]:
    s=least_squares(res,th0,x_scale=[0.05,0.05,0.1,0.1])
    print('fit dcx,dcy,c2scale,mu',s.x,'rms rel',np.sqrt(np.mean(s.fun**2)))
pr,md=predict(s.x,X,True); e=np.abs(pr-X[:,6:9])/X[:,13:14]
print('sliding frac',md.mean(),'rel err mean per comp',e.mean(0),'95pct',np.percentile(e,95,axis=0))
print('baseline (dcx=dcy=0,sc=1) rms',np.sqrt(np.mean(res([0,0,1,s.x[3]])**2)))
for mu in [0.2,0.3,0.4,0.5,0.7,1.0,5]:
    print(' mu',mu,'rms',np.sqrt(np.mean(res([s.x[0],s.x[1],s.x[2],mu])**2)).round(4))
np.save('probe_dyn_X.npy',X)
for sd in np.unique(X[:,15]):
    m=X[:,15]==sd; Xs=X[m]
    s2=least_squares(lambda th:((predict(th,Xs)-Xs[:,6:9])/Xs[:,13:14]).ravel(),s.x)
    print('seed',int(sd),'n',m.sum(),'fit',s2.x.round(4),'dims',Xs[0,9:12].round(3),'centroid cy,c2',np.round(centroid(*Xs[0,9:12]),4))
# robot kinematic check
rd=np.linalg.norm(D[:,13:15]-D[:,6:8]-D[:,8:10],axis=1); print('max |robot disp - u|',rd.max())
# per-speed error
for v in [0.01,0.02,0.03,0.0499]:
    m=np.isclose(X[:,13],v); print('v',v,'rel err',e[m].mean(0).round(4))
