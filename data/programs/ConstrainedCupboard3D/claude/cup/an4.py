import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
R=(slice(30,170),slice(250,400))
for a,b in [(0,1),(0,7),(1,7)]:
    d=np.abs(A[a][R]-A[b][R]).sum(2)
    print(a,b,'maxdiff',d.max(),'npix>12',(d>12).sum())
