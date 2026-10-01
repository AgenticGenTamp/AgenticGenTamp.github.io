import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
a=A[0]
for y,x,t in [(60,320,'topplate'),(100,300,'backpanel?'),(140,285,'shelf'),(95,270,'darkslot'),(100,260,'floor'),(60,250,'wall')]:
    print(t,(y,x),a[y,x])
a1=A[1]
print('--seed1--')
for y,x,t in [(60,320,'topplate'),(100,300,'back'),(140,300,'shelf'),(100,340,'?'),(150,355,'robot?'),(100,275,'floor')]:
    print(t,(y,x),a1[y,x])
