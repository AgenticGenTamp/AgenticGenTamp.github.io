from cal_trial import *
t=trial(0,'part0',0,0,0.05); d=t['d']
for g in np.linspace(-1,1,21):
    d.step(act(grip=g))
    if d.r['ga']<1: print('released at',g); break
print('ga after sweep',d.r['ga'])
b=d.r['base']; q=d.r['q']
for tgt in [[0.313,0.289,0.25],[0.3,0.0,0.25]]+[[0.3,0.0,z] for z in np.arange(0.24,0.16,-0.002)]:
    qs,_,_=ik(tgt,q,base=b); ok=d.goto_q(qs); q=d.r['q']
    if not ok: print('rejected at',tgt); break
print('part',np.round(part(d.obs,'part0')[0][:3],4))
d.step(act(grip=1.0)); print('open on rack ga',d.r['ga'],'pga',part(d.obs,'part0')[1],'part',np.round(part(d.obs,'part0')[0][:3],4))
q=d.r['q']; qs,_,_=ik([0.3,0,0.3],q,base=b); d.goto_q(qs); print('after lift part',np.round(part(d.obs,'part0')[0][:3],4),'ga',d.r['ga'])
