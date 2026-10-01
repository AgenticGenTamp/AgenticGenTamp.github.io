from lib import *
import sys
seed=int(sys.argv[1]); a=float(sys.argv[2]); step=float(sys.argv[3]) if len(sys.argv)>3 else 0.005
e=E(seed); V=e.obs[9:11].copy(); th=e.obs[11]
u=np.array([-np.cos(th),-np.sin(th)]); nn=np.array([np.sin(th),-np.cos(th)])
S=V+a*u+0.4*nn
face=np.arctan2(-nn[1],-nn[0])
print('goto',e.goto(S[0],S[1],face,verbose=False))
for i in range(400):
    p=e.obs[:2].copy(); e.st([-nn[0]*step,-nn[1]*step,0,0,0])
    if np.allclose(p,e.obs[:2],atol=1e-7): break
rel=e.obs[:2]-V; print('block a=%.4f b=%.4f'%(rel@u,rel@nn), 'robot',e.obs[:9])
h0=e.obs[9:12].copy(); r0=e.obs[:3].copy()
e.st([0,0,0,0,1]); print('after vac', e.obs[:9], 'hook d', e.obs[9:12]-h0)
for d in [nn, -nn, u, -u]:
    h0=e.obs[9:12].copy(); r0=e.obs[:3].copy()
    e.st([d[0]*0.01,d[1]*0.01,0,0,1]); print('mv',d,'robot d',e.obs[:3]-r0,'hook d', e.obs[9:12]-h0)
