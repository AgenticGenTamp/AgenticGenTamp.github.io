from cal_trial import *
t=trial(0,'part0',0,0,0.05); d=t['d']; print('ga',t['ga'])
pp0,_=part(d.obs,'part0'); b=d.r['base']
q=d.r['q']; qs,_,_=ik([pp0[0],pp0[1],pp0[2]+0.1],q,base=b); print('lift',d.goto_q(qs))
for g in [1.0,1.0,1.0,0.6,0.51]:
    d.step(act(grip=g)); print('open',g,'ga',d.r['ga'],'finger',d.r['finger'],part(d.obs,'part0')[1])
# lower back to table and open
q=d.r['q']; 
for z in np.arange(pp0[2]+0.1,pp0[2]+0.045,-0.01):
    qs,_,_=ik([pp0[0],pp0[1],z],q,base=b); ok=d.goto_q(qs); q=d.r['q']
print('lowered',ok,np.round(part(d.obs,'part0')[0][:3],4))
d.step(act(grip=1.0)); print('open on table ga',d.r['ga'],part(d.obs,'part0')[1])
