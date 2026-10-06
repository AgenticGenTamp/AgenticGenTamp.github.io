import time, numpy as np, approach as ap
k=ap._Kin.get()
t=time.time(); k.canon(0.3); k.canon(0.035); print('base',time.time()-t)
t=time.time(); n=0
for z in np.arange(0.03,0.45,0.004):
    for tau in np.arange(0.3,0.9,0.1):
        q=k.canon(z,tau); n+=1
        p,R=k.fk(q, np.zeros(3))
        zs=R[:,2]
        if abs(p[0]-ap.REACH)>2e-3 or abs(p[2]-round(z/0.004)*0.004)>2e-3 or abs(q[0])>0.3: print('bad',z,tau,p,q.round(2))
print(n,'solves',time.time()-t)
