import numpy as np
from probe_vis_lib import *
# fit a, R from theta-threshold data
dat=np.load('thr_theta.npy')
O=np.array([2.1176,1.923]); u=np.array([-0.2911,-0.9567]); u/=np.linalg.norm(u)
def resid(a,R):
    s=0
    for th,d in dat:
        e=np.array([np.cos(th),np.sin(th)])
        v=d*u+a*e
        s+=(np.linalg.norm(v)-R)**2
    return s
best=None
for a in np.arange(-0.2,0.201,0.001):
    for R in np.arange(1.90,2.10,0.001):
        r=resid(a,R)
        if best is None or r<best[0]: best=(r,a,R)
print("range-circle fit: rms=%.4f  a=%.3f (camera offset along heading) R2d=%.3f"%(np.sqrt(best[0]/len(dat)),best[1],best[2]))
a_fit,R_fit=best[1],best[2]
# now occlusion fit on grid data (theta=pi during grid run)
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); Og=np.array([L['objective0']['x'],L['objective0']['y']])
obst=[(n,np.array([L[n]['x'],L[n]['y']]),L[n]['half_x'],L[n]['half_y'],L[n]['z'],L[n]['half_z']) for n in L if n.startswith('obstacle')]
env.close()
recs=np.load('vis_grid_39.npy')
e_th=np.array([np.cos(np.pi),np.sin(np.pi)])
def predict(P,H,pad,R,a,zt=0.201):
    cam=P+a*e_th
    d=np.linalg.norm(Og-cam)
    if d>R: return False
    uu=(Og-cam)/d
    for n,q,hx,hy,z,hz in obst:
        w=q-cam; t=np.dot(w,uu); perp=abs(w[0]*uu[1]-w[1]*uu[0])
        if not (0<t<d): continue
        if perp>hx+pad: continue
        hray=H+(zt-H)*(t/d)
        if hray<z+hz: return False
    return True
best=None
for H in np.arange(0.30,1.21,0.01):
  for pad in np.arange(0.0,0.21,0.005):
      err=sum(predict(np.array([x,y]),H,pad,R_fit,a_fit)!=bool(v) for x,y,v in recs)
      if best is None or err<best[0]: best=(err,H,pad)
print("grid occlusion fit with cam offset a=%.3f: err=%d/%d  H=%.2f pad=%.3f"%(a_fit,best[0],len(recs),best[1],best[2]))
H,pad=best[1],best[2]
for x,y,v in recs:
    p=predict(np.array([x,y]),H,pad,R_fit,a_fit)
    if p!=bool(v):
        cam=np.array([x,y])+a_fit*e_th; d=np.linalg.norm(Og-cam)
        print("   ERR (%.1f,%.1f) obs=%d pred=%d d_cam=%.3f"%(x,y,v,p,d))
