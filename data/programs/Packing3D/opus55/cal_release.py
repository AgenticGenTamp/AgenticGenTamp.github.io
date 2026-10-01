from cal_trial import *
t=trial(0,'part0',0,0,0.05); d=t['d']; print('ga',t['ga'])
pp0,_=part(d.obs,'part0'); b=d.r['base']
q=d.r['q']; qs,_,_=ik([pp0[0],pp0[1],pp0[2]+0.2],q,base=b); print('lift',d.goto_q(qs))
p1,_=part(d.obs,'part0'); print('part lifted',np.round(p1[:3],4))
d.step(act(grip=1.0)); r=d.r; p2,pga=part(d.obs,'part0'); print('after open: ga',r['ga'],'pga',pga,'part',np.round(p2[:3],4),'gtf',r['gtf'][:3])
for i in range(3): d.step(act()); 
p3,_=part(d.obs,'part0'); print('after 3 noop',np.round(p3[:3],4))
q=d.r['q']; qs,_,_=ik([pp0[0]-0.1,pp0[1],pp0[2]+0.25],q,base=b); print('move away',d.goto_q(qs))
p4,_=part(d.obs,'part0'); print('after arm move',np.round(p4[:3],4))
# free-space close then move to part with grip=0
d=Driver(0); d.step(act(grip=-1.0)); print('free close ga',d.r['ga'],d.r['finger'])
pp,_=part(d.obs,'part0'); b=d.r['base']; q=d.r['q']
for z in np.arange(pp[2]+0.15,pp[2]+0.049,-0.01):
    qs,_,_=ik([pp[0],pp[1],z],q,base=b); d.goto_q(qs,grip=0.0); q=d.r['q']
print('arrived w/o open, ga',d.r['ga'])
d.step(act()); print('noop ga',d.r['ga'])
d.step(act(grip=-1.0)); print('close ga',d.r['ga'])
d.step(act(grip=-1.0)); print('close again ga',d.r['ga'], 'gtf', np.round(d.r['gtf'][:3],4))
