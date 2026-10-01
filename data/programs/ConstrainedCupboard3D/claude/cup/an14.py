import numpy as np
exec(open('/sandbox/cup/an.py').read().split('for s in')[0])
def runs(mask,off=0):
    out=[];st=None
    for i,v in enumerate(mask):
        if v and st is None: st=i
        if not v and st is not None:
            if i-1-st>=1: out.append((off+st,off+i-1))
            st=None
    if st is not None and len(mask)-1-st>=1: out.append((off+st,off+len(mask)-1))
    return out
files={'s0':'/sandbox/cup/s0.ppm','s7':'/sandbox/cup/s7.ppm',
 'oc3_seed1':'/sandbox/cup/state_seed1_oc3_seed1.ppm','oc3_seed2':'/sandbox/cup/state_seed2_oc3_seed2.ppm',
 'oc3_seed3':'/sandbox/cup/state_seed3_oc3_seed3.ppm','oc3_seed11':'/sandbox/cup/state_seed11_oc3_seed11.ppm'}
imgs={k:load(v).astype(int) for k,v in files.items()}
ref=imgs['s0'][40:160,260:380]
for k,a in imgs.items():
    print('===',k,'cupdiff_vs_s0',(np.abs(a[40:160,260:380]-ref).sum(2)>12).sum())
    lum=a.mean(2); R,G,B=a[...,0],a[...,1],a[...,2]
    cup=((R-G)<=8)&((G-B)>=9)&(R>60)
    sig={}
    for x in range(265,376):
        br=(lum[68:158,x]>=125)&cup[68:158,x]
        sig[x]=tuple(runs(br,68))
    cur=None;st=None
    for x in range(265,377):
        v=sig.get(x,())
        if v!=cur:
            if cur and x-st>=4: print('   x %3d-%3d (w%2d): %s'%(st,x-1,x-st,list(cur)))
            cur=v;st=x
