import numpy as np, base64, zlib, time
from approach import Sim, Planner, CARRY_ARM
s=Sim(1); P=Planner(s)
X0,Y0,RES,NX,NY,NT = -1.3,-1.4,0.05,53,57,16
occ=np.zeros((NT,NY,NX),bool)
t=time.time()
for k in range(NT):
    yaw=-np.pi+2*np.pi*k/NT
    for j in range(NY):
        for i in range(NX):
            x=X0+RES*i; y=Y0+RES*j
            if abs(x)>1.05 or abs(y)>1.25: continue
            q=np.concatenate([[x,y,yaw],CARRY_ARM])
            occ[k,j,i]=P.table_collision(q)
print('time',time.time()-t, occ.sum())
open('occ_b64.txt','w').write(base64.b64encode(zlib.compress(np.packbits(occ).tobytes(),9)).decode())
