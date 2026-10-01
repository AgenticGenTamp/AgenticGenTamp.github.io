import numpy as np,glob
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
LV={(71,87):'H',(72,87):'H',(90,111):'M',(89,111):'M',(112,133):'L',(91,111):'M'}
for f in sorted(glob.glob('/sandbox/cup/state_seed*_oc*.ppm'))+['/sandbox/cup/s0.ppm','/sandbox/cup/s1.ppm','/sandbox/cup/s7.ppm']:
    a=load(f).astype(int); lum=a.mean(2); R,G,B=a[...,0],a[...,1],a[...,2]
    cup=((R-G)<=8)&((G-B)>=9)&(R>60)
    sig={}
    for x in range(262,380):
        br=(lum[68:158,x]>=125)&cup[68:158,x]
        sig[x]=tuple(runs(br,68))
    cur=None;st=None;segs=[]
    for x in range(262,381):
        v=sig.get(x,())
        if v!=cur:
            if cur and x-st>=4: segs.append((st,x-1,cur))
            cur=v;st=x
    lab=[]
    for a_,b_,rr in segs:
        top=rr[0]
        lab.append(('%.0f'%((a_+b_)/2), LV.get(top,str(top))))
    print('%-42s'%f.split('/')[-1], ' '.join('%s@%s'%(l,c) for c,l in lab))
