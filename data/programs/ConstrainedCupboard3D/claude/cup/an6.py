import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
for s in [0,1,7]:
    lum=A[s].mean(2)
    reg=lum[45:160,255:395]
    dark=(reg<45)
    print('seed',s,'dark col counts x=255..394:')
    cc=dark.sum(0)
    print(' '.join('%d:%d'%(255+i,c) for i,c in enumerate(cc) if c>0))
