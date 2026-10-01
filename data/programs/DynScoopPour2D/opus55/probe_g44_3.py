from probe_g44_2 import *
import sys
a,arm,gap,depth,xout,dy=[float(v) for v in sys.argv[1].split(',')]; seed=int(sys.argv[2]) if len(sys.argv)>2 else 44
p=P(seed); h0=p.h(); approach(p,a,arm,gap,xout)
log=[]; ok=descend(p,0.5-depth,dy=dy,log=log)
pre=p.h(); k=close(p); r=p.r(); h=p.h()
print(sys.argv[1],seed,'h0x',h0['x'],'OK' if k else 'FAIL','pre',pre['x'],'r',r,'h',h,flush=True)
