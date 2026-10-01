import sys, pickle
from cal_util import *
from ik import ik
from fk import fk
rng=np.random.default_rng(1)
d=Driver(0); pn='part0'
pp,_=part(d.obs,pn); b=d.r['base']; q=d.r['q']
qs,_,_=ik([pp[0]-0.1,pp[1],0.26],q,base=b,yaw=0.0); d.goto_q(qs,grip=1.0)
d.step(act(grip=-1.0)); print('ga',d.r['ga'])
data=[]
def rec():
    r=d.r; p,pga=part(d.obs,pn); data.append(dict(q=r['q'],base=r['base'],gtf=r['gtf'],part=p,ga=r['ga']))
q=d.r['q']; qs,_,_=ik([pp[0]-0.1,pp[1],0.40],q,base=b,yaw=0.0); print('lift ok',d.goto_q(qs)); rec()
q0=d.r['q'].copy(); b0=d.r['base'].copy()
for k in range(40):
    bt=b0+np.r_[rng.uniform(-0.15,0.0),rng.uniform(-0.15,0.15),rng.uniform(-0.5,0.5)]
    okb=d.goto_base(bt)
    qt=q0+rng.uniform(-0.3,0.3,7); ok=d.goto_q(qt)
    rec()
print('n',len(data),'rej',d.nrej, 'bases',np.round(np.array([s['base'] for s in data])[:6],3))
pickle.dump(data,open('cal_data2.pkl','wb'))
