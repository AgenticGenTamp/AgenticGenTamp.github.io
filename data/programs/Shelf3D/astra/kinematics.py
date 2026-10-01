import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
# Standard Gen3 joint frame transforms.
OFFSETS=[(0,0,.15643),(0,.005375,-.12838),(0,-.21038,-.006375),(0,.006375,-.21038),(0,-.20843,-.006375),(0,0,-.10593),(0,-.10593,0)]
ANGLES=[np.pi,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2]
ROTS=[Rotation.from_euler('x',a).as_matrix() for a in ANGLES]
HOME=np.array([0,-.34906585,np.pi,-2.5481807,0,-.8726646,np.pi/2])
def fk(q, height=.3, tip=.16):
 p=np.array([0.,0.,height]);R=np.eye(3)
 for t,A,a in zip(OFFSETS,ROTS,q):
  p+=R@t
  c,s=np.cos(a),np.sin(a)
  R=R@A@np.array([[c,-s,0],[s,c,0],[0,0,1]])
 p+=R@np.array([0,0,-tip])
 return p,R
if __name__=='__main__':
 print(fk(HOME))
