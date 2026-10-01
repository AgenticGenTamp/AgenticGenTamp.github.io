from lib import *
import sys
def trial(seed,a,side,step=0.005,vacfirst=False,rot=0.0):
    e=E(seed); V=e.obs[9:11].copy(); th=e.obs[11]
    u=np.array([-np.cos(th),-np.sin(th)]); nn=np.array([np.sin(th),-np.cos(th)])
    b0=0.05+0.4 if side==1 else -0.4
    S=V+a*u+b0*nn
    if not(0.12<S[0]<3.38 and 0.12<S[1]<1.13): return 'skip'
    face=np.arctan2(-side*nn[1],-side*nn[0])+rot
    if not e.goto(S[0],S[1],face,verbose=False): return 'gotofail'
    v=1 if vacfirst else 0
    for i in range(400):
        p=e.obs[:2].copy(); e.st([-side*nn[0]*step,-side*nn[1]*step,0,0,v])
        if np.allclose(p,e.obs[:2],atol=1e-7): break
        if vacfirst and not np.allclose(e.obs[9:12],V.tolist()+[th],atol=1e-6): return 'pushed?'
    rel=e.obs[:2]-V; b=rel@nn
    e.st([0,0,0,0,1])
    res=[]
    for d in [nn,-nn,u,-u]:
        h0=e.obs[9:12].copy(); r0=e.obs[:2].copy()
        e.st([d[0]*0.01,d[1]*0.01,0,0,1])
        rm=not np.allclose(r0,e.obs[:2]); hm=not np.allclose(h0,e.obs[9:12])
        res.append('G' if hm else ('R' if rm else 'x'))  # G grasp-follow, R robot moved hook not, x blocked
    return 'b=%.3f %s'%(b,''.join(res))
if __name__=='__main__':
    s0,s1=int(sys.argv[1]),int(sys.argv[2]); kw=eval('dict(%s)'%(sys.argv[3] if len(sys.argv)>3 else ''))
    for seed in range(s0,s1):
        print(seed,[trial(seed,a,side,**kw) for a in [0.4,0.7,1.0] for side in [1,-1]])
