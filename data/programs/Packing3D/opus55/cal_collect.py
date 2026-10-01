import sys, pickle
from cal_util import *
from ik import ik
from fk import fk
rng=np.random.default_rng(0)
d=Driver(0); pn='part0'
pp,_=part(d.obs,pn); b=d.r['base']; q=d.r['q']
qs,_,_=ik([pp[0]-0.1,pp[1],0.26],q,base=b,yaw=0.0); d.goto_q(qs,grip=1.0)
d.step(act(grip=-1.0)); print('ga',d.r['ga'])
data=[]
def rec():
    r=d.r; p,pga=part(d.obs,pn); data.append(dict(q=r['q'],base=r['base'],gtf=r['gtf'],part=p,ga=r['ga']))
rec()
# lift
q=d.r['q']; qs,_,_=ik([pp[0]-0.1,pp[1],0.35],q,base=b,yaw=0.0); print('lift ok',d.goto_q(qs)); rec()
print('part after lift',part(d.obs,pn)[0][:3])
q0=d.r['q'].copy()
for k in range(60):
    qt=q0+rng.uniform(-0.4,0.4,7)
    ok=d.goto_q(qt)
    if ok: rec()
print('n',len(data),'rej',d.nrej)
pickle.dump(data,open('cal_data.pkl','wb'))
