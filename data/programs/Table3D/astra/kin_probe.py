import numpy as np
from scipy.spatial.transform import Rotation as R
origins=[([0,0,.15643],[np.pi,0,0]),([0,.005375,-.12838],[np.pi/2,0,0]),([0,-.21038,-.006375],[-np.pi/2,0,0]),([0,.006375,-.21038],[np.pi/2,0,0]),([0,-.20843,-.006375],[-np.pi/2,0,0]),([0,.00017505,-.10593],[np.pi/2,0,0]),([0,-.10593,-.00017505],[-np.pi/2,0,0])]
def fk(q):
 t=np.eye(4)
 for (p,r),v in zip(origins,q):
  m=np.eye(4);m[:3,3]=p;m[:3,:3]=R.from_euler('xyz',r).as_matrix()@R.from_euler('z',v).as_matrix();t=t@m
 m=np.eye(4);m[:3,3]=[0,0,-.061525];m[:3,:3]=R.from_euler('x',np.pi).as_matrix();t=t@m
 return t
if __name__=='__main__':
 for q2 in [-.9,-.35,0,.5]:
  q=[0,q2,-np.pi,-2.5,0,q2+2.5-np.pi,np.pi/2]
  print(q2,fk(q))
