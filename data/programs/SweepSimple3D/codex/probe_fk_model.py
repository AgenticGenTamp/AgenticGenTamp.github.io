"""Offline approximate Kinova Gen3 FK from published URDF joint origins."""
import numpy as np
from scipy.spatial.transform import Rotation

ORIGINS = [
 ((0, 0, .15643), (np.pi, 0, 0)),
 ((0, .005375, -.12838), (np.pi/2, 0, np.pi)),
 ((0, -.21038, -.006375), (-np.pi/2, 0, np.pi)),
 ((0, .006375, -.21038), (np.pi/2, 0, np.pi)),
 ((0, -.20843, -.006375), (-np.pi/2, 0, np.pi)),
 ((0, 0, -.10593), (np.pi/2, 0, np.pi)),
 ((0, -.10593, 0), (-np.pi/2, 0, np.pi)),
]

def tf(xyz=(0,0,0), rpy=(0,0,0)):
 t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz;return t

def fk(q):
 t=np.eye(4)
 for (xyz,rpy),a in zip(ORIGINS,q):
  t=t@tf(xyz,rpy)@tf(rpy=(0,0,a))
 return t@tf((0,0,-.13),(np.pi,0,0))

if __name__=='__main__':
 for q in ([0,-.349,np.pi,-2.548,0,-.873,np.pi/2],
           [0,.32,2.37,-2.57,.02,-.8,np.pi/2]):
  print(np.round(fk(q),3))
