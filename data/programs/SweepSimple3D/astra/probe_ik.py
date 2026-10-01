from probe_fk import fk
from scipy.optimize import least_squares
import numpy as np
for x,z in [(.6,.22),(.6,.3),(.7,.22),(.6,.01),(.7,.01)]:
 def fun(v):
  q2,q4=v;q=[0,q2,np.pi,q4,0,q2-q4-np.pi,np.pi/2];t=fk(q);p=t[:3,3]+.12*t[:3,2]+[.2,0,.35];return p[[0,2]]-[x,z]
 sol=least_squares(fun,[1.6,-1],bounds=([-2.2,-2.5],[2.24,2.5]));q2,q4=sol.x
 print('TARGET',x,z,'Q',q2,q4,q2-q4-np.pi,'ERR',fun(sol.x))
