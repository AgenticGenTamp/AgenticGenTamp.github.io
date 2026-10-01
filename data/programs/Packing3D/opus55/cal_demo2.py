"""Grasp both parts (yaw fixed 0) and place them in rack at specified part poses."""
import sys
from cal_util import *
from ik import ik, ik_pose, tcp_for_part_pose, handle_offset
seed=int(sys.argv[1]); goals={'part0':(0.3,0.07),'part1':(float(sys.argv[2]),float(sys.argv[3]))}
d=Driver(seed); b=d.r['base']; q=d.r['q']
def move(tgt=None,M=None,grip=0.0):
    global q
    qs,pe,_=(ik(tgt,q,base=b,yaw=0.0) if M is None else ik_pose(M,q,base=b)); ok=d.goto_q(qs,grip=grip); q=d.r['q']; return ok
for pn,(gx,gy) in goals.items():
    P=d.obs.get_object_from_name(pn); pp,_=part(d.obs,pn)
    tt=d.obs.get(P,'triangle_type') if 'Triangle' in str(P) else None
    h=pp[:2]+handle_offset(None,tt)
    move([h[0],h[1],pp[2]+0.15],grip=1.0)
    for z in np.arange(0.12,0.049,-0.01): move([h[0],h[1],pp[2]+z])
    d.step(act(grip=-1.0)); gtf=d.r['gtf'].copy(); print(pn,'grasp',d.r['ga'],np.round(gtf[:3],4))
    move([h[0],h[1],0.30])
    z=0.30-0.05+0.0; M=tcp_for_part_pose([gx,gy,0.25],pp[3:7],gtf); move(M=M)
    ok=True; pz=0.25
    while ok and pz>0.05:
        pz-=0.002; ok=move(M=tcp_for_part_pose([gx,gy,pz],pp[3:7],gtf))
    o,r,te,tr,info=d.env.step(act(grip=1.0)); d.obs=o
    p,_=part(d.obs,pn); print(pn,'placed at',np.round(p[:3],4),'quat',np.round(p[3:7],3),'ga',d.r['ga'],'r',r,'term',te)
    move([gx,gy,0.30])
print('steps',d.steps,'rej',d.nrej)
