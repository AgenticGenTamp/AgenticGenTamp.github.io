"""Numerical Kinova arm model calibrated through black-box grasp experiments."""
import numpy as np
from math import sin,cos,pi

def fk(q,tool=.12):
 T=np.eye(4)
 for a,p,v in zip([pi,pi/2,-pi/2,pi/2,-pi/2,pi/2,-pi/2],[(0,0,.15643),(0,.005375,-.12838),(0,-.21038,-.006375),(0,.006375,-.21038),(0,-.20843,-.006375),(0,0,-.10593),(0,-.10593,0)],q):
  c,s=cos(a),sin(a);cv,sv=cos(v),sin(v)
  A=np.eye(4); A[:3,:3]=np.array([[1,0,0],[0,c,-s],[0,s,c]])@np.array([[cv,-sv,0],[sv,cv,0],[0,0,1]]); A[:3,3]=p;T=T@A
 T[:3,3]+=T[:3,2]*(-.061525-tool)
 T[:3,1:3]*=-1
 return T
