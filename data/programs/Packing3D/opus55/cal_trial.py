from cal_util import *
from ik import ik
from fk import fk
def trial(seed, pn, dx, dy, dz, yaw=0.0, d=None, verbose=False):
    """Move TCP to part_center+(dx,dy,dz) with z-down & yaw, close, return info."""
    if d is None: d=Driver(seed)
    pp,_=part(d.obs,pn); b=d.r['base']; q=d.r['q']
    tgt=np.array([pp[0]+dx,pp[1]+dy,pp[2]+dz])
    reached=True
    for z in list(np.arange(tgt[2]+0.12,tgt[2],-0.01))+[tgt[2]]:
        qs,pe,re=ik([tgt[0],tgt[1],z],q,base=b,yaw=yaw)
        if not d.goto_q(qs,grip=1.0): reached=False; break
        q=d.r['q']
    M,_=fk(d.r['q'],base=b)
    d.step(act(grip=-1.0)); r=d.r; _,pga=part(d.obs,pn)
    return dict(reached=reached, tcp=M[:3,3]-pp[:3], ga=r['ga'], pga=pga, gtf=r['gtf'], d=d)
if __name__=='__main__':
    import sys
    for dz in [0.1,0.08,0.06,0.05,0.04,0.03,0.02,0.01,0.0]:
        t=trial(0,'part0',0,0,dz)
        print(f"dz={dz} reached={t['reached']} tcp_rel={np.round(t['tcp'],3)} ga={t['ga']} gtf={np.round(t['gtf'][:3],4)}")
