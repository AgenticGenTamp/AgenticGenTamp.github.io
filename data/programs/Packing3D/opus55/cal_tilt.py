from cal_util import *
from ik import ik, down_R
from fk import fk
from scipy.spatial.transform import Rotation as Rot
for ang in [0.2,0.4,0.8]:
    d=Driver(0); pp,_=part(d.obs,'part0'); b=d.r['base']; q=d.r['q']
    Rt=Rot.from_rotvec([0,ang,0]).as_matrix()@down_R(0)
    tgt=pp[:3]+np.array([0,0,0.05])
    for z in np.arange(0.12,-0.001,-0.02):
        qs,pe,re=ik(tgt+[0,0,z],q,base=b,R_target=Rt); ok=d.goto_q(qs,grip=1.0); q=d.r['q']
        if not ok: break
    M,_=fk(d.r['q'],base=b)
    d.step(act(grip=-1.0))
    print(f'tilt={ang} reached={ok} tcp_rel={np.round(M[:3,3]-pp[:3],3)} zaxis={np.round(M[:3,2],2)} ga={d.r["ga"]} gtf={np.round(d.r["gtf"],3)}')
