"""Approximate Kinova Gen3 chain from published dimensions; pure numpy."""
import numpy as np

_ORIGINS = [
 ((0.,0.,.15643),np.pi),
 ((0.,.005375,-.12838),np.pi/2),
 ((0.,-.21038,-.006375),-np.pi/2),
 ((0.,.006375,-.21038),np.pi/2),
 ((0.,-.20843,-.006375),-np.pi/2),
 ((0.,0.,-.10593),np.pi/2),
 ((0.,-.10593,0.),-np.pi/2),
]

def rx(t):
 c,s=np.cos(t),np.sin(t)
 return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rz(t):
 c,s=np.cos(t),np.sin(t)
 return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def forward(q, base=(0.,0.,0.), mount_height=.25, tool_length=.16, points=False):
 """World grasp midpoint; mount_height and tool_length need empirical calibration."""
 p=np.array([base[0],base[1],mount_height],dtype=float)
 r=rz(base[2]); chain=[p.copy()]
 for angle,(xyz,roll) in zip(q,_ORIGINS):
  p=p+r@xyz
  r=r@rx(roll)@rz(angle)
  chain.append(p.copy())
 p=p+r@np.array([0.,0.,-(.061525+tool_length)])
 r=r@rx(np.pi)
 chain.append(p.copy())
 return (p,r,np.array(chain)) if points else (p,r)

def planar_ik(x, z, mount_height=.50, tool_length=.16, guess=None):
 """Elbow-up, downward-facing grasp in base x/z plane."""
 from scipy.optimize import least_squares
 q=np.array([0.,-.349066,np.pi,-2.54818,0.,-.872665,np.pi/2])
 def fun(v):
  q[[1,3,5]]=v
  p,r=forward(q,mount_height=mount_height,tool_length=tool_length)
  return np.array([p[0]-x,p[2]-z,r[0,2]])
 result=least_squares(fun, q[[1,3,5]] if guess is None else guess,
  bounds=([-2.2,-2.65,-2.2],[2.2,-.05,2.2]), max_nfev=70)
 q[[1,3,5]]=result.x
 return q.copy()
