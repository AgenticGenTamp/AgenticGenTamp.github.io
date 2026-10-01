import numpy as np
from scipy.spatial.transform import Rotation
P = np.array([[0,0,.15643],[0,.005375,-.12838],[0,-.21038,-.006375],[0,.006375,-.21038],[0,-.20843,-.006375],[0,.00017505,-.10593],[0,-.10593,-.00017505],[0,0,-.061525]])
RX=[np.pi,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi]
R=[Rotation.from_euler('x',x).as_matrix() for x in RX]
def fk(q):
 r=np.eye(3);p=np.zeros(3)
 for i in range(8):
  p=p+r@P[i];r=r@R[i]
  if i<7:
   c,s=np.cos(q[i]),np.sin(q[i]);r=r@np.array([[c,-s,0],[s,c,0],[0,0,1]])
 return p,r
if __name__=='__main__':
 q=np.deg2rad([0,-20,180,-146,0,-50,90]);p,r=fk(q);print(p,r, 'tip',p+r@np.array([0,0,.12]))
