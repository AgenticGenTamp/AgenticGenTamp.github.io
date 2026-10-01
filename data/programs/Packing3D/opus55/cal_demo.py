"""End-to-end: grasp part, carry to rack, lower until contact, release."""
import sys
from cal_util import *
from ik import ik
from fk import fk
seed=int(sys.argv[1]); pn=sys.argv[2]; gx,gy=float(sys.argv[3]),float(sys.argv[4])
d=Driver(seed); b=d.r['base']; q=d.r['q']
pp,_=part(d.obs,pn); P=d.obs.get_object_from_name(pn)
hoff=np.array([0.0333,0.0333]) if 'triangle_type' in [] else np.zeros(2)
try:
    if d.obs.get(P,'triangle_type')==1.0: hoff=np.array([0.0333,0.0333])
except Exception: pass
h=pp[:2]+hoff
def move(tgt,grip=0.0):
    global q
    qs,pe,_=ik(tgt,q,base=b); ok=d.goto_q(qs,grip=grip); q=d.r['q']; return ok
move([h[0],h[1],pp[2]+0.15],grip=1.0)             # above handle, gripper open
for z in np.arange(0.12,0.049,-0.01): move([h[0],h[1],pp[2]+z])  # descend to TCP 5cm above part center
d.step(act(grip=-1.0)); print('grasp ga',d.r['ga'],'gtf',np.round(d.r['gtf'][:3],4))
gz=d.r['gtf'][2]
move([h[0],h[1],0.30])                               # lift
move([gx+hoff[0],gy+hoff[1],0.30])                   # carry (TCP above goal)
z=0.30; ok=True
while ok and z>0.05:                                 # lower in 2mm steps until contact rejection
    z-=0.002; ok=move([gx+hoff[0],gy+hoff[1],z])
p,_=part(d.obs,pn); print('lowest part z',round(p[2],4))
o,r,te,tr,info=d.env.step(act(grip=1.0)); d.obs=o
print('release ga',d.r['ga'],'reward',r,'term',te)
move([gx+hoff[0],gy+hoff[1],0.30]); p,_=part(d.obs,pn); print('part after arm retreat',np.round(p[:3],4),'steps',d.steps,'rej',d.nrej)
