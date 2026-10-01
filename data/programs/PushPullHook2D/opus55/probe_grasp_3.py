from lib import *
import sys
# gap tolerance and rotation tolerance on seed 4 (inner side works) and 2 (outer)
def run(seed,a,side,gap=0.0,rot=0.0,armback=0.0):
    e=E(seed); V=e.obs[9:11].copy(); th=e.obs[11]
    u=np.array([-np.cos(th),-np.sin(th)]); nn=np.array([np.sin(th),-np.cos(th)])
    b0=0.05+0.4 if side==1 else -0.4
    S=V+a*u+b0*nn
    face=np.arctan2(-side*nn[1],-side*nn[0])
    if not e.goto(S[0],S[1],face,verbose=False): return 'gotofail'
    for step in [0.005,0.001,0.0002]:
      for i in range(400):
        p=e.obs[:2].copy(); e.st([-side*nn[0]*step,-side*nn[1]*step,0,0,0])
        if np.allclose(p,e.obs[:2],atol=1e-8): break
    b=(e.obs[:2]-V)@nn
    if gap: e.st([side*nn[0]*gap,side*nn[1]*gap,0,0,0])
    if armback: e.st([0,0,0,-armback,0])
    if rot:
        r=e.obs[2]; e.st([0,0,rot,0,0])
        if abs(e.obs[2]-r)<1e-6: return 'b=%.4f rotblocked'%b
    e.st([0,0,0,0,1])
    res=''
    for d in [nn,-nn,u,-u]:
        h0=e.obs[9:12].copy(); r0=e.obs[:2].copy()
        e.st([d[0]*0.01,d[1]*0.01,0,0,1])
        rm=not np.allclose(r0,e.obs[:2]); hm=not np.allclose(h0,e.obs[9:12])
        res+='G' if hm else ('R' if rm else 'x')
    return 'b=%.4f %s'%(b,res)
if __name__=="__main__":
 for seed,a,side in [(4,0.7,1),(2,0.7,-1)]:
    for g in [0,0.001,0.002,0.005,0.01,0.02]: print(seed,'gap',g,run(seed,a,side,gap=g))
    for r in [0.05,-0.05,0.2,-0.2]: print(seed,'rot',r,run(seed,a,side,rot=r))
