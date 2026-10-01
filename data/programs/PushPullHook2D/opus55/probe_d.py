from lib import *
import sys
seed=int(sys.argv[1]); a=float(sys.argv[2]); mode=sys.argv[3]
e=E(seed); V=e.obs[9:11].copy(); th=e.obs[11]
u=np.array([-np.cos(th),-np.sin(th)]); nn=np.array([np.sin(th),-np.cos(th)])
S=V+a*u+0.4*nn
face=np.arctan2(-nn[1],-nn[0])
e.goto(S[0],S[1],face,verbose=False)
for i in range(80):
    p=e.obs[:2].copy(); e.st([-nn[0]*0.005,-nn[1]*0.005,0,0,0])
    if np.allclose(p,e.obs[:2],atol=1e-7): break
rel=e.obs[:2]-V; print('block b=%.3f'%(rel@nn))
if mode=='arm':
    for i in range(10):
        p=e.obs[4]; e.st([0,0,0,0.005,0])
        if abs(p-e.obs[4])<1e-7: break
    print('arm',e.obs[4])
if mode=='back':
    e.st([nn[0]*0.004,nn[1]*0.004,0,0,0]);
    e.st([0,0,0,0.01,0]); print('arm',e.obs[4])
h0=e.obs[9:12].copy()
e.st([0,0,0,0,1]); e.st([nn[0]*0.01,nn[1]*0.01,0,0,1]); print('hook moved', e.obs[9:12]-h0)
