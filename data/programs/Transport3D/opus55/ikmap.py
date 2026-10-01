import numpy as np, time
from kin import ik, fk_arm
Rdown=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
home=np.array([0,-0.35,-3.142,-2.5,0,-0.87,1.571])
t=time.time(); n=0
for z in [-0.2,-0.045,0.2,0.355,0.45]:
    row=[]
    for r in np.arange(0.1,0.85,0.05):
        q,ep,er=ik(np.array([r,0,z]),Rdown,home); n+=1
        row.append('%d'%(ep<1e-3 and er<1e-2))
    print(z,''.join(row))
print((time.time()-t)/n,'s per ik')
q,ep,er=ik(np.array([0.5,0,0.355]),Rdown,home); print(np.round(q,2))
q,ep,er=ik(np.array([0.5,0,-0.2]),Rdown,home); print(np.round(q,2))
