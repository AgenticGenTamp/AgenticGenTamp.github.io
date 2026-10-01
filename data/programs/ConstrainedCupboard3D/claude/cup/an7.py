import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
for s in [0,1,7]:
    lum=A[s].mean(2)
    # cupboard bright top plate: find rows/cols where lum>145
    reg=lum[35:170,240:410]
    b=reg>140
    rs=b.sum(1); cs=b.sum(0)
    print('seed',s)
    print('  bright rows:', ' '.join('%d:%d'%(35+i,c) for i,c in enumerate(rs) if c>0))
    print('  bright cols:', ' '.join('%d:%d'%(240+i,c) for i,c in enumerate(cs) if c>2))
