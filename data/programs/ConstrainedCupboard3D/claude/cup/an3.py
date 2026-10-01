import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
a=load('cup/s0.ppm').astype(int)
med=np.median(a,axis=1,keepdims=True)
d=np.abs(a-med).sum(2)>20
ys,xs=np.nonzero(d)
print('fg bbox y',ys.min(),ys.max(),'x',xs.min(),xs.max(),'n',d.sum())
cs=d.sum(0)
print('col profile (every 10):')
for i in range(0,640,10): print(i,cs[i], end='; ')
print()
rs=d.sum(1)
print('row profile (every 10):')
for i in range(0,480,10): print(i,rs[i], end='; ')
