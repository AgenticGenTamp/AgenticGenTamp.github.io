import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
for s in [0,1,7]:
    a=A[s]; R,G,B=a[...,0],a[...,1],a[...,2]
    m=((R-G)<=8)&((G-B)>=9)&(R>60)
    m[:,:200]=False; m[:,430:]=False; m[200:,:]=False
    ys,xs=np.nonzero(m)
    print('seed',s,'bbox y',ys.min(),ys.max(),'x',xs.min(),xs.max(),'n',m.sum())
    for y in range(ys.min(),ys.max()+1):
        r=np.nonzero(m[y])[0]
        if len(r): print('   y%3d x%3d-%3d w%3d n%3d'%(y,r.min(),r.max(),r.max()-r.min()+1,len(r)))
