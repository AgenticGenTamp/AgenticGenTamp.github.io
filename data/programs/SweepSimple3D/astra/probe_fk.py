import numpy as np
from scipy.spatial.transform import Rotation as R
origins=[([0,0,.15643],[np.pi,0,0]),([0,.005375,-.12838],[np.pi/2,0,0]),([0,-.21038,-.006375],[-np.pi/2,0,0]),([0,.006375,-.21038],[np.pi/2,0,0]),([0,-.20843,-.006375],[-np.pi/2,0,0]),([0,0,-.10593],[np.pi/2,0,0]),([0,-.10593,0],[-np.pi/2,0,0]),([0,0,-.061525],[np.pi,0,0])]
def fk(q):
 t=np.eye(4)
 for i,(p,r) in enumerate(origins):
  o=np.eye(4);o[:3,:3]=R.from_euler('xyz',r).as_matrix();o[:3,3]=p;t=t@o
  if i<7:
   o=np.eye(4);o[:3,:3]=R.from_euler('z',q[i]).as_matrix();t=t@o
 return t
for q in [[0,-.349,np.pi,-2.548,0,-.873,np.pi/2],[0,1.7,np.pi,-1.5,0,-.873,np.pi/2],[0,2.24,np.pi,.3,0,-.873,np.pi/2],[0,2.24,np.pi,.3,0,-2,np.pi/2],[0,2.24,np.pi,1.,0,-2,np.pi/2]]:
 t=fk(q);print(q, np.round(t[:3,3],3),np.round(t[:3,2],3))
