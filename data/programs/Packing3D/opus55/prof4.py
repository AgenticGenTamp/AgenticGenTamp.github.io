import numpy as np, ik10
from ik import down_R
from scipy.optimize import least_squares, minimize
c0=np.array([-0.12,0,0, 0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2])
tp=np.array([0.3,0.0,0.2])
idx=np.arange(3,10)
def full(x):
    c=c0.copy(); c[idx]=x; return c
def r1(x):
    c=full(x); return np.concatenate([ik10._pose_res(c,tp,None,True)*10, 0.02*(c-c0)[idx]])
for meth in ['lm','trf']:
    s=least_squares(r1,c0[idx],method=meth,xtol=1e-10,ftol=1e-10,max_nfev=300)
    print(meth, np.round(full(s.x)-c0,2), s.status, s.nfev, np.linalg.norm(ik10._pose_res(full(s.x),tp,None,True)))
n=7
x1=s.x; t1=np.max(np.abs(full(x1)-c0)); z0=np.concatenate([x1,[t1]])
cons=[{'type':'eq','fun':lambda z: ik10._pose_res(full(z[:n]),tp,None,True)},
      {'type':'ineq','fun':lambda z: np.concatenate([z[n]-(z[:n]-c0[idx]), z[n]+(z[:n]-c0[idx])])}]
s2=minimize(lambda z:z[n], z0, method='SLSQP', constraints=cons, options={'maxiter':60,'ftol':1e-7})
print(s2.status, s2.message, s2.nit, np.round(full(s2.x[:n])-c0,3), s2.x[n], np.linalg.norm(ik10._pose_res(full(s2.x[:n]),tp,None,True)))
