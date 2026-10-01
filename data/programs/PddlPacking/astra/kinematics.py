"""PR2 tool kinematics calibrated against observed rigid grasp transforms."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
AXES=[2,1,0,1,0,1,0]
OFF=[(0,0,0),(.1,0,0),(0,0,0),(.4,0,0),(0,0,0),(.321,0,0),(0,0,0)]
Q0=np.array([.67717,-.34313,1.2,-1.46688,1.24223,-1.95443,2.22254])
def fk(q,h=.990675):
 p=np.array([-.05,.188,h]); r=np.eye(3)
 for a,ax,off in zip(q,AXES,OFF):
  p+=r@off
  c=np.cos(a); s=np.sin(a)
  if ax==0: rr=np.array([[1,0,0],[0,c,-s],[0,s,c]])
  elif ax==1: rr=np.array([[c,0,s],[0,1,0],[-s,0,c]])
  else: rr=np.array([[c,-s,0],[s,c,0],[0,0,1]])
  r=r@rr
 return p+r@np.array([.18,0,0]),r
DOWN=Rotation.from_euler('y',np.pi/2).as_matrix()
def ik(p,h=.990675,q0=Q0,rot=DOWN):
 # A weak posture cost keeps successive IK solutions on the same arm branch.
 def err(q):
  x,r=fk(q,h)
  return np.r_[4*(x-p),Rotation.from_matrix(rot.T@r).as_rotvec(),.005*(q-q0)]
 return least_squares(err,q0,max_nfev=70).x
