from probe_dyn_lib import *
from probe_dyn_b import exp
for v in [0.01,0.0499]:
  for a in [10,20,30,45,60]:
    t=np.radians(a)
    exp(f'e stem bottom oblique {a}',[-np.sin(t)*0.25,-1.2-np.cos(t)*0.25],[np.sin(t),np.cos(t)],v,int(0.35/v),verbose=(v==0.01 and a in(20,45)))
  for a in [20,45]:
    t=np.radians(a)
    exp(f'e bar top x0 oblique {a}',[-np.sin(t)*0.3,0.3*np.cos(t)],[np.sin(t),-np.cos(t)],v,int(0.35/v))
