import numpy as np
from env_client import make_env
env=make_env()
rows=[]
for s in range(200):
    o,_=env.reset(seed=s); rows.append(o)
A=np.array(rows)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for name,i in [('rx',0),('ry',1),('rth',2),('hx',9),('hy',10),('hth',11),('bx',20),('by',21),('tx',29),('ty',30)]:
    print(name, A[:,i].min(), A[:,i].max())
d=A[:,29:31]-A[:,20:22]
print('dist', np.linalg.norm(d,axis=1).min(), np.linalg.norm(d,axis=1).max())
ang=np.degrees(np.arctan2(d[:,1],d[:,0]))
print('angle hist', np.histogram(ang,bins=12,range=(-180,180))[0])
const=[i for i in range(38) if np.ptp(A[:,i])<1e-6]; print('const idx',const)
