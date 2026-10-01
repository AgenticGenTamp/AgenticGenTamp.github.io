import time,numpy as np
exec(open('fitbox.py').read().split("names=[")[0])
o=diffmask('drawer0'); ys,xs=np.nonzero(o)
win=(xs.min()-40,xs.max()+40,ys.min()-40,ys.max()+40)
ow=o[win[2]:win[3],win[0]:win[1]]
t=time.time(); print(loss(np.array([0.65,0.0,0.25,0.15,0.15,0.08]),ow,win)); print('one eval',time.time()-t)
from scipy.optimize import minimize
t=time.time()
r=minimize(loss,np.array([0.65,0.0,0.25,0.15,0.15,0.08]),args=(ow,win,'sym'),method='Powell',options={'maxiter':2000,'xtol':1e-3,'ftol':1e-3})
print('powell',time.time()-t, r.fun, np.round(r.x,3), r.nfev)
