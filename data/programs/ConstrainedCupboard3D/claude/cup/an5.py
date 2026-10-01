import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
a=A[0]
sub=a[40:160,255:395]
# histogram of luminance
lum=sub.mean(2)
h,_=np.histogram(lum,bins=np.arange(0,260,10))
print(list(zip(range(0,250,10),h)))
