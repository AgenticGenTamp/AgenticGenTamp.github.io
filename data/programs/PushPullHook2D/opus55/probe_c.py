from lib import *
import sys
seed=int(sys.argv[1])
for a in [0.35,0.6,0.9]:
  for side in [1,-1]:
    e=E(seed); V=e.obs[9:11].copy(); th=e.obs[11]
    u=np.array([-np.cos(th),-np.sin(th)]); nn=np.array([np.sin(th),-np.cos(th)])
    b0=0.05+0.3 if side==1 else -0.3
    S=V+a*u+b0*nn
    if not(0.12<S[0]<3.38 and 0.12<S[1]<1.13): print(a,side,'skip'); continue
    face=np.arctan2(-side*nn[1],-side*nn[0])
    if not e.goto(S[0],S[1],face,verbose=False): print(a,side,'goto fail'); continue
    for i in range(80):
        p=e.obs[:2].copy(); e.st([-side*nn[0]*0.005,-side*nn[1]*0.005,0,0,0])
        if np.allclose(p,e.obs[:2],atol=1e-7): break
    rel=e.obs[:2]-V
    print(a,side,'block at a=%.3f b=%.3f'%(rel@u, rel@nn))
