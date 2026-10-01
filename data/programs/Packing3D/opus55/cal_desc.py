import sys
from cal_util import *
from ik import ik
from fk import fk
seed=int(sys.argv[1]); pname=sys.argv[2]; yaw=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
d=Driver(seed)
pp,_=part(d.obs,pname); print('part',pp[:3])
b=d.r['base']; q=d.r['q']
z=0.40
while z>-0.2:
    qs,pe,re=ik([pp[0],pp[1],z],q,base=b,yaw=yaw)
    ok=d.goto_q(qs,grip=1.0)
    r=d.r; M,_=fk(r['q'],base=b)
    d.step(act(grip=-1.0)); r2=d.r
    _,pga=part(d.obs,pname)
    print(f'z={z:.3f} ik_err={pe:.1e} ok={ok} ee={np.round(M[:3,3],3)} finger={r2["finger"]} ga={r2["ga"]} pga={pga}')
    if r2['ga']>0 or not ok: print('gtf',r2['gtf']); break
    d.step(act(grip=1.0))
    q=d.r['q']; z-=0.01
print("steps",d.steps,"rej",d.nrej); d.save()
