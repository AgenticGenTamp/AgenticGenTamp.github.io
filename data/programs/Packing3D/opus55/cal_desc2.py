import sys
from cal_util import *
from ik import ik
from fk import fk
seed=int(sys.argv[1]); pname=sys.argv[2]; yaw=float(sys.argv[3]); dx=float(sys.argv[4]); dy=float(sys.argv[5])
d=Driver(seed)
pp,_=part(d.obs,pname)
b=d.r['base']; q=d.r['q']
z=0.26; last=None
while z>0.0:
    qs,pe,re=ik([pp[0]+dx,pp[1]+dy,z],q,base=b,yaw=yaw)
    ok=d.goto_q(qs,grip=1.0)
    if not ok: break
    last=z; q=d.r['q']; z-=0.002
M,_=fk(d.r['q'],base=b)
d.step(act(grip=-1.0)); r2=d.r; _,pga=part(d.obs,pname)
print(f'yaw={yaw:.2f} dxy=({dx},{dy}) lowest_ok_z={last:.3f} ee={np.round(M[:3,3],4)} ga={r2["ga"]} pga={pga} gtf={np.round(r2["gtf"],4)}')
d.save()
