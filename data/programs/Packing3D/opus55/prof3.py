import numpy as np
from ik import ik, down_R, wrap_near
c0=np.array([-0.12,0,0, 0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2])
for tgt in [[0.3,0.3,0.15],[0.2,-0.33,0.15],[0.3,0.0,0.2]]:
    q,pe,_=ik(tgt,c0[3:],base=c0[:3],yaw=None)
    q=wrap_near(q,c0[3:])
    print(tgt, np.round(q-c0[3:],2), pe)
