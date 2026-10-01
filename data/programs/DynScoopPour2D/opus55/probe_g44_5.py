from probe_g44_4 import *
import sys
# args: prefix(0/1) a depth seed
pre,a,depth,seed=int(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
p=P(seed); goto2(p,y=2.2); goto2(p,x=3.0); goto2(p,y=1.4)
if pre:
    attempt(p,0.16,0.4,0.25,0.08,3.499,0.005); r=p.r(); goto2(p,y=r['y']+0.25)
h0=p.h()
good,tix=attempt(p,a,0.4,0.25,depth,3.499,0.00125); hp=p.h()
k=close(p); r=p.r(); h=p.h()
print(sys.argv[1:],'h0',h0['x'],'pre',hp['x'],'tix %.4f'%tix,'OK' if k else 'FAIL',k,'r',r,'h',h,flush=True)
