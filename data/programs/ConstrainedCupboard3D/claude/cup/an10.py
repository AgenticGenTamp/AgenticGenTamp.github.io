import numpy as np
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
def runs(mask):
    out=[];st=None
    for i,v in enumerate(mask):
        if v and st is None: st=i
        if not v and st is not None: out.append((st,i-1)); st=None
    if st is not None: out.append((st,len(mask)-1))
    return out
for s in [0,1,7]:
    a=A[s]; lum=a[...,0]*0+a.mean(2)
    R,G,B=a[...,0],a[...,1],a[...,2]
    cup=((R-G)<=8)&((G-B)>=9)&(R>60)
    print('=== seed',s)
    x0,x1=(267,373) if s!=1 else (292,348)
    for x in range(x0,x1+1):
        col=lum[68:158,x]
        br=(col>=125)&cup[68:158,x]
        print(' x%3d'%x, [(68+a_,68+b_) for a_,b_ in runs(br) if b_-a_>=1])
