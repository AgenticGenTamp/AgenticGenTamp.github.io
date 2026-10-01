import numpy as np
from scipy.spatial.transform import Rotation as R
s=np.load('initial_0.npy')
origins=[(0,0,.15643),(0,.005375,-.12838),(0,-.21038,-.006375),(0,.006375,-.21038),(0,-.20843,-.006375),(0,0,-.10593),(0,-.10593,0),(0,0,-.061525)]
rps=[np.pi,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi]
T=np.eye(4)
for i,(p,r) in enumerate(zip(origins,rps)):
 A=np.eye(4);A[:3,3]=p;A[:3,:3]=R.from_euler('x',r).as_matrix();T=T@A
 if i<7:
  A=np.eye(4);A[:3,:3]=R.from_euler('z',s[128+i]).as_matrix();T=T@A
 print(i, np.round(T[:3,3],4))
print('ee',T,'tool',T[:3,3]+T[:3,2]*.12)
