from probe_g44_4 import *
import sys
def att2(p,a,arm,gap,depth,xout,dy):
    d,n,to,ti=geo(a,arm,gap); Rx=min(3.3,xout-to[0])
    goto2(p,y=1.4); goto2(p,x=min(3.0,Rx-0.1)); goto2(p,th=-np.pi/2+a,arm=arm,gap=gap); goto2(p,x=Rx)
    descend(p,0.5-depth,dy=dy)
    r=p.r(); return r['x']+ti[0], r
pre,a,arm,depth,seed=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),int(sys.argv[5])
p=P(seed); goto2(p,y=2.2); goto2(p,x=3.0); goto2(p,y=1.4)
if pre:
    attempt(p,0.16,0.4,0.25,0.08,3.499,0.005); r=p.r(); goto2(p,y=r['y']+0.25)
h0=p.h()
tix,r0=att2(p,a,arm,0.25,depth,3.499,0.00125); hp=p.h()
k=close(p); r=p.r(); h=p.h()
print(sys.argv[1:],'h0',h0['x'],'pre',hp,'tix %.4f'%tix,'r0',r0,'OK' if k else 'FAIL',k,'r',r,'h',h,flush=True)
