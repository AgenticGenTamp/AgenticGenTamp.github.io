from cal_trial import *
t=trial(0,'part0',0,0,0.05); d=t['d']; print('ga',t['ga'],'finger',d.r['finger'])
for g in [0.0,0.3,-0.3,-1.0,-1.0,1.0,0.0]:
    d.step(act(grip=g)); print('grip',g,'ga',d.r['ga'],'finger',d.r['finger'])
pp0,_=part(d.obs,'part0'); b=d.r['base']; q=d.r['q']
# carry to above rack
for tgt in [[pp0[0],pp0[1],0.25],[0.3,0.0,0.25]]+[[0.3,0.0,z] for z in np.arange(0.24,0.17,-0.01)]:
    qs,_,_=ik(tgt,q,base=b); ok=d.goto_q(qs); q=d.r['q']
    print('tgt',np.round(tgt,3),ok,'part',np.round(part(d.obs,'part0')[0][:3],4))
    if not ok: break
d.step(act(grip=1.0)); print('open above rack ga',d.r['ga'],'part',np.round(part(d.obs,'part0')[0][:3],4))
