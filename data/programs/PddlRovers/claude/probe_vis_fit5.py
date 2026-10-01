import numpy as np
dat=np.load('thr_theta.npy')
O=np.array([2.1176,1.923]); u=np.array([-0.2911,-0.9567]); u/=np.linalg.norm(u)
def resid(a,b,R):
    s=0
    for th,d in dat:
        e=np.array([np.cos(th),np.sin(th)]); n=np.array([-np.sin(th),np.cos(th)])
        s+=(np.linalg.norm(d*u+a*e+b*n)-R)**2
    return s
best=None
for a in np.arange(-0.3,0.301,0.005):
  for b in np.arange(-0.3,0.301,0.005):
    for R in np.arange(1.90,2.10,0.005):
        r=resid(a,b,R)
        if best is None or r<best[0]: best=(r,a,b,R)
print("fit rms=%.4f a=%.3f b=%.3f R=%.3f"%(np.sqrt(best[0]/len(dat)),best[1],best[2],best[3]))
for th,d in dat:
    e=np.array([np.cos(th),np.sin(th)]); n=np.array([-np.sin(th),np.cos(th)])
    print("  th=%+.2f d=%.4f  model_r=%.4f"%(th,d,np.linalg.norm(d*u+best[1]*e+best[2]*n)))
