import numpy as np
from approach import *
from approach import _get_model
k=Kin(_get_model())
q0=np.array([0.,-0.349,3.142,-2.548,0.,-0.873,1.571])
rng=np.random.default_rng(0)
B=np.array([1.5,0,np.pi])
def near(q,ref):
    q=q.copy()
    for j in (0,2,4,6): q[j]=ref[j]+((q[j]-ref[j]+np.pi)%(2*np.pi)-np.pi)
    return q
res=[]
for n in range(5000):
    bx=rng.uniform(1.2,1.5); by=rng.uniform(-0.6,0.6); yaw=np.pi+rng.uniform(-1.3,1.3); psi=rng.uniform(0.0,0.4)
    cs=[bx+np.cos(yaw)*0.29-np.sin(yaw)*s for s in (0.27,-0.27)]
    if min(cs)<0.97: continue
    q,ok=k.ik(np.array([bx,by,yaw]),q0,np.array([0.934+0.09,0,0.355]),R_front(psi),iters=80)
    if not ok: continue
    q=near(q,q0)
    if np.any(q<LIM[:,0]) or np.any(q>LIM[:,1]): continue
    c1=np.abs(q-q0).max()
    best2=9
    for phi in (np.pi/2, 0.0):
        qp,ok2=k.ik(B,q,np.array([0.8,0,0.55]),R_grasp(0.2,phi),iters=80)
        if ok2:
            qp=near(qp,q)
            if np.all(qp>=LIM[:,0]) and np.all(qp<=LIM[:,1]):
                best2=min(best2,np.abs(qp-q).max())
    res.append((c1+best2,c1,best2,bx,by,yaw,psi))
res.sort()
for r in res[:12]: print(np.round(r,3))
# current
q,ok=k.ik(np.array([1.26,-0.3,np.pi]),q0,np.array([0.934+0.09,0,0.355]),R_front(0.3),iters=80); q=near(q,q0)
qp,ok2=k.ik(B,q,np.array([0.8,0,0.55]),R_grasp(0.2,np.pi/2),iters=80); qp=near(qp,q)
print('current',np.abs(q-q0).max(),np.abs(qp-q).max(),ok,ok2)
