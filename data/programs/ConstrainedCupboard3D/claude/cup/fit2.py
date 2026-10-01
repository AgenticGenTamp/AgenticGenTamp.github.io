import sys; sys.path.insert(0,"/sandbox")
import numpy as np, json
from scipy.optimize import least_squares, brentq
P=[];UV=[]
exec(open('/sandbox/cup/fitcam.py').read().split("obs=[]")[0].replace("from scipy.optimize import least_squares",""))
obs=[]
for k,f in files.items():
    a=load('/sandbox/cup/%s.ppm'%f).astype(int)
    bl=blobs(a); rods=D[k]['rods']
    if len(bl)!=len(rods): continue
    ys=[r[1] for r in rods]
    if min(abs(ys[i]-ys[j]) for i in range(len(ys)) for j in range(i+1,len(ys)))<0.12: continue
    bl2=sorted(bl,key=lambda t:t[1]); rd=sorted(rods,key=lambda r:-r[1])
    for b,r in zip(bl2,rd): obs.append((r[0],r[1],0.03,b[1],b[0],k))
print('rod obs',len(obs))
Pr=np.array([[o[0],o[1],o[2]] for o in obs]); UVr=np.array([[o[3],o[4]] for o in obs])
DIVY=np.array([0.3,0.2,0.1,-0.1,-0.2,-0.3]); DIVU=np.array([271.5,288.5,304.5,334.5,350.5,367.5])
BOTY=np.array([0.25,0.15,0.05,-0.05,-0.15,-0.25]); BOTV=152.5
def rod3(rv):
    t=np.linalg.norm(rv)
    if t<1e-9: return np.eye(3)
    k=rv/t; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
def proj(p,pts):
    C=p[:3]; Rm=rod3(p[3:6]); f=p[6]
    pc=(Rm@(np.asarray(pts,float)-C).T).T
    return np.stack([320+f*pc[:,0]/pc[:,2], 240+f*pc[:,1]/pc[:,2]],1)
W=3.0
def res(p):
    xf=p[7]
    r1=(proj(p,Pr)-UVr).ravel()
    r2=W*(proj(p,np.stack([np.full(6,xf),DIVY,np.full(6,0.25)],1))[:,0]-DIVU)
    r3=W*(proj(p,np.stack([np.full(6,xf),BOTY,np.zeros(6)],1))[:,1]-BOTV)
    return np.concatenate([r1,r2,r3])
p0=np.concatenate([np.load('/sandbox/cup/cam.npy'),[1.85]])
s=least_squares(res,p0,method='lm',max_nfev=50000); p=s.x
print('rms',np.sqrt(np.mean(s.fun**2)),'cam',p[:3],'f=%.1f'%p[6],'xf=%.3f'%p[7],'fovy=%.1fdeg'%(2*np.degrees(np.arctan(240/p[6]))))
print('rod resid px', np.round((proj(p,Pr)-UVr),1).tolist())
print('div resid', np.round(proj(p,np.stack([np.full(6,p[7]),DIVY,np.full(6,0.25)],1))[:,0]-DIVU,2))
print('bot resid', np.round(proj(p,np.stack([np.full(6,p[7]),BOTY,np.zeros(6)],1))[:,1]-BOTV,2))
np.save('/sandbox/cup/cam2.npy',p)
def zof(v,xf,y=0.0): return brentq(lambda z: proj(p,[[xf,y,z]])[0][1]-v,-1.0,4.0)
xf=p[7]
for lbl,v in [('cup bottom edge',152.5),('base-plate front lip',150.5),('LOW-shelf front',133.5),('MID-shelf front',111.5),('HIGH-shelf front',87.5),('top-plate front(near) edge',67.0),('top-plate far edge',47.0)]:
    print('  %-28s v=%5.1f -> z=%.3f m (at x=%.2f)'%(lbl,v,zof(v,xf),xf))
print('  --- same rows evaluated at x=2.0 and x=2.15 ---')
for lbl,v in [('base',150.5),('LOW',133.5),('MID',111.5),('HIGH',87.5),('top',67.0)]:
    print('  %-6s v=%5.1f -> z(x=2.00)=%.3f  z(x=2.15)=%.3f'%(lbl,v,zof(v,2.0),zof(v,2.15)))
