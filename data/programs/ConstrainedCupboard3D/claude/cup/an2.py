import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
for a,b in [(0,1),(0,7),(1,7)]:
    d=np.abs(A[a]-A[b]).sum(2)>12
    ys,xs=np.nonzero(d)
    print(a,b,'ndiff',d.sum(),'bbox y',ys.min(),ys.max(),'x',xs.min(),xs.max())
    # per-column profile
    colsum=d.sum(0); rowsum=d.sum(1)
    print('  cols>0 ranges:', end=' ')
    idx=np.nonzero(colsum)[0]
    runs=[];st=idx[0];pr=idx[0]
    for i in idx[1:]:
        if i>pr+3: runs.append((st,pr)); st=i
        pr=i
    runs.append((st,pr)); print(runs)
