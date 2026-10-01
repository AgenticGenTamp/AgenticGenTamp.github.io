import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
d=np.abs(A[0]-A[7]).sum(2)
ys,xs=np.nonzero(d>12)
print('0vs7 full-image ndiff',(d>12).sum(),'bbox y',ys.min(),ys.max(),'x',xs.min(),xs.max())
m=(d>12); m2=m[30:170,240:410]
ys,xs=np.nonzero(m2)
print('in cup region n',m2.sum(),'y',ys.min()+30,ys.max()+30,'x',xs.min()+240,xs.max()+240)
# where exactly
import collections
print(collections.Counter([(y+30)//10*10 for y in ys]))
