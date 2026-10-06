import numpy as np, time
exec(open('simcore.py').read()); exec(open('fk_part.py').read())
s=Sim(2); ff=FastFK(s)
rng=np.random.default_rng(0)
for t in range(5):
    q=rng.uniform(-1,1,10); q[3:]=np.clip(q[3:],s.lo[3:],s.hi[3:])
    pp,pq=s.fk(q); F=ff.fk(q)
    print(np.abs(F[:3,3]-pp).max(), np.abs(F[:3,:3]-np.array(p.getMatrixFromQuaternion(pq)).reshape(3,3)).max())
    _,J=ff.jac(q); _,_,J2=s.jac(q); print('J',np.abs(J-J2).max())
t=time.time()
for i in range(1000): ff.jac(q)
print((time.time()-t)/1000)
