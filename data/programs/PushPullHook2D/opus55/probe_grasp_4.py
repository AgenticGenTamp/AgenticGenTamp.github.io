from lib import *
import sys
def run(seed,a,side,rot=0.0,vacfirst=False,arm=False):
    e=E(seed); V=e.obs[9:11].copy(); th=e.obs[11]; H=e.obs[9:12].copy()
    u=np.array([-np.cos(th),-np.sin(th)]); nn=np.array([np.sin(th),-np.cos(th)])
    b0=0.05+0.4 if side==1 else -0.4
    if arm: b0=0.05+0.17 if side==1 else -0.17
    S=V+a*u+b0*nn
    face=np.arctan2(-side*nn[1],-side*nn[0])+rot
    if not e.goto(S[0],S[1],face,verbose=False): return 'gotofail'
    v=1 if vacfirst else 0
    if arm:
        for i in range(100):
            p=e.obs[4]; e.st([0,0,0,0.005,v])
            if abs(p-e.obs[4])<1e-8 or not np.allclose(H,e.obs[9:12]): break
        b=e.obs[4]
    else:
      for step in [0.005]:
        for i in range(400):
          p=e.obs[:2].copy(); e.st([-side*nn[0]*step,-side*nn[1]*step,0,0,v])
          if np.allclose(p,e.obs[:2],atol=1e-8) or not np.allclose(H,e.obs[9:12]): break
      b=(e.obs[:2]-V)@nn
    if not np.allclose(H,e.obs[9:12]): return 'b=%.4f hook moved during approach'%b
    e.st([0,0,0,0,1])
    res=''
    for d in [nn,-nn,u,-u]:
        h0=e.obs[9:12].copy(); r0=e.obs[:2].copy()
        e.st([d[0]*0.01,d[1]*0.01,0,0,1])
        rm=not np.allclose(r0,e.obs[:2]); hm=not np.allclose(h0,e.obs[9:12])
        res+='G' if hm else ('R' if rm else 'x')
    return 'b/arm=%.4f %s'%(b,res)
if __name__=="__main__":
  for seed,a,side in [(4,0.7,1),(2,0.7,-1)]:
    for r in [0.1,0.3,0.6,1.0,1.4]: print(seed,'rot',r,run(seed,a,side,rot=r))
    print(seed,'vacfirst',run(seed,a,side,vacfirst=True))
    print(seed,'arm',run(seed,a,side,arm=True))
    print(seed,'arm vacfirst',run(seed,a,side,arm=True,vacfirst=True))
