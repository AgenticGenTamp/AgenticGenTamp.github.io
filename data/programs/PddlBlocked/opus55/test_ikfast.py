import numpy as np, time
from kin import ik_single, ik_single_old, fk_world
from kinpy import fk
rng=np.random.default_rng(0)
q=np.array([0.39,0.33,0.0,-1.52,2.72,-1.22,-2.99]); b=np.array([3.8,0.1,0.2])
p1,R1=fk_world(b,q); p2,cols,_,_,_=fk(b,q); print('fk match',np.abs(p1-np.array(p2)).max(),np.abs(R1-np.array(cols).T).max())
T=(4.5,0,0.3,0.6)
for fb in [False,True]:
  e1=[];e2=[]; t1=0;t2=0
  for k in range(30):
    tgt=np.array([4.3+0.2*rng.random(), -0.2+0.5*rng.random(), 0.8+0.1*rng.random()]); ang=rng.uniform(-1,1); d=np.array([np.cos(ang),np.sin(ang),0])
    s=time.time(); _,_,ea=ik_single_old(tgt,d,b,q,free_base=fb,elbow_min=0.82,table=T); t1+=time.time()-s
    s=time.time(); _,_,eb=ik_single(tgt,d,b,q,free_base=fb,elbow_min=0.82,table=T); t2+=time.time()-s
    e1.append(ea); e2.append(eb)
  print('free',fb,'old t',round(t1,2),'succ',np.mean(np.array(e1)<5e-3),' new t',round(t2,2),'succ',np.mean(np.array(e2)<5e-3))
from kin import ik
e=[];t=time.time()
rng=np.random.default_rng(0)
for k in range(30):
    tgt=np.array([4.3+0.2*rng.random(), -0.2+0.5*rng.random(), 0.8+0.1*rng.random()]); ang=rng.uniform(-1,1); d=np.array([np.cos(ang),np.sin(ang),0])
    _,_,eb=ik(tgt,d,b,q,free_base=True,elbow_min=0.82,table=T); e.append(eb)
print('wrapper free', time.time()-t, np.mean(np.array(e)<5e-3))
