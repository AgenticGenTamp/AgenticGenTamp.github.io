from cal_trial import *
import sys
tx,ty=float(sys.argv[1]),float(sys.argv[2])
GZ=float(sys.argv[3]) if len(sys.argv)>3 else 0.05; t=trial(0,'part0',0,0,GZ); print('gtf',np.round(t['gtf'][:3],4)); d=t['d']
b=d.r['base']; q=d.r['q']
for tgt in [[0.313,0.289,0.25],[tx,ty,0.25]]+[[tx,ty,z] for z in np.arange(0.2,0.10,-0.001)]:
    qs,_,_=ik(tgt,q,base=b); ok=d.goto_q(qs,grip=0.0); q=d.r['q']
    if tgt[2]<0.2: d.step(act(grip=1.0))
    p=part(d.obs,'part0')[0]
    if d.r['ga']<1 or not ok: print('tgt',np.round(tgt,3),'ok',ok,'ga',d.r['ga'],'part',np.round(p[:3],4)); break
