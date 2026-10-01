from cal_trial import *
import sys
tx,ty=float(sys.argv[1]),float(sys.argv[2])
t=trial(0,'part0',0,0,0.05); d=t['d']; b=d.r['base']; q=d.r['q']
for tgt in [[0.313,0.289,0.3],[tx,ty,0.3]]+[[tx,ty,z] for z in np.arange(0.2,0.10,-0.001)]:
    qs,_,_=ik(tgt,q,base=b); ok=d.goto_q(qs,grip=0.0); q=d.r['q']
    if not ok: break
p=part(d.obs,'part0')[0]; print(f'({tx},{ty}) lowest part z={p[2]:.4f} rejected_next={not ok}')
d.step(act(grip=1.0)); print(' open ga',d.r['ga'])
