import numpy as np,collections
exec(open('cup/an.py').read().split('for s in')[0])
A={s:load('cup/s%d.ppm'%s).astype(int) for s in [0,1,7]}
def runs(mask,off=0):
    out=[];st=None
    for i,v in enumerate(mask):
        if v and st is None: st=i
        if not v and st is not None:
            if i-1-st>=1: out.append((off+st,off+i-1))
            st=None
    if st is not None and len(mask)-1-st>=1: out.append((off+st,off+len(mask)-1))
    return out
for s in [0,1,7]:
    a=A[s]; lum=a.mean(2); R,G,B=a[...,0],a[...,1],a[...,2]
    cup=((R-G)<=8)&((G-B)>=9)&(R>60)
    x0,x1=(265,375) if s!=1 else (290,350)
    sig={}
    for x in range(x0,x1+1):
        br=(lum[68:158,x]>=125)&cup[68:158,x]
        sig[x]=tuple(runs(br,68))
    # group consecutive columns with same signature
    print('=== seed',s)
    cur=None;st=None
    for x in range(x0,x1+2):
        v=sig.get(x,())
        if v!=cur:
            if cur: print('  x %3d-%3d (w%2d): %s'%(st,x-1,x-st,list(cur)))
            cur=v;st=x
