import numpy as np
from scipy.spatial.transform import Rotation as R
q=[0,-.35,-np.pi,-2.5,0,-.87,np.pi/2]
params=[([0,0,.15643],np.pi),([0,.005375,-.12838],np.pi/2),([0,-.21038,-.006375],-np.pi/2),([0,.006375,-.21038],np.pi/2),([0,-.20843,-.006375],-np.pi/2),([0,0,-.10593],np.pi/2),([0,-.10593,0],-np.pi/2)]
t=np.eye(4)
for (xyz,rx),qi in zip(params,q):
 a=np.eye(4);a[:3,3]=xyz;a[:3,:3]=R.from_euler('x',rx).as_matrix()@R.from_euler('z',qi).as_matrix();t=t@a
print(t)
for z in [-.061525,-.12,-.18,-.2]: print(z,t[:3,3]+t[:3,2]*z)
