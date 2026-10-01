import numpy as np,glob
exec(open('/sandbox/cup/an.py').read().split('for s in')[0])
for f in sorted(glob.glob('/sandbox/cup/state_seed*_oc*.ppm'))+['/sandbox/cup/s0.ppm','/sandbox/cup/s1.ppm','/sandbox/cup/s7.ppm']:
    a=load(f).astype(int); R,G,B=a[...,0],a[...,1],a[...,2]
    m=((R-G)<=8)&((G-B)>=9)&(R>60); m[:,:250]=False; m[:,400:]=False; m[160:]=False
    ys,xs=np.nonzero(m)
    # widest row
    w=[(np.nonzero(m[y])[0].max()-np.nonzero(m[y])[0].min()+1) if m[y].any() else 0 for y in range(40,160)]
    print('%-32s top=%d bot=%d xmin=%d xmax=%d width@y47=%d maxw=%d'%(f.split('/')[-1],ys.min(),ys.max(),xs.min(),xs.max(),w[7],max(w)))
