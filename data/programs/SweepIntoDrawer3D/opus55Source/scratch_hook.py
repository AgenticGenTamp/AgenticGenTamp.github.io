import numpy as np
from approach import *
from approach import _get_model
k=Kin(_get_model())
q0=np.array([0.,-0.349,3.142,-2.548,0.,-0.873,1.571])
B=np.array([1.5,0,np.pi])
def near(q,ref):
    q=q.copy()
    for j in (0,2,4,6): q[j]=ref[j]+((q[j]-ref[j]+np.pi)%(2*np.pi)-np.pi)
    return q
def Rhook(a):
    z=np.array([-np.sin(a),0,-np.cos(a)]); x=np.array([0,1.,0]); y=np.cross(z,x); return np.column_stack([x,y,z])
res=[]
rng=np.random.default_rng(1)
for n in range(3000):
    a=rng.uniform(0.4,1.1); bx=rng.uniform(1.22,1.45); by=rng.uniform(-0.5,0.5)
    dirv=np.array([-np.sin(a),0,-np.cos(a)])
    site=np.array([0.912,0,0.333])-0.02*dirv
    q,ok=k.ik(np.array([bx,by,np.pi]),q0,site,Rhook(a),iters=80)
    if not ok: continue
    q=near(q,q0)
    if np.any(q<LIM[:,0]) or np.any(q>LIM[:,1]): continue
    c1=np.abs(q-q0).max()
    qp,ok2=k.ik(B,q,np.array([0.8,0,0.55]),R_grasp(0.2,np.pi/2),iters=80)
    if not ok2: continue
    qp=near(qp,q); c2=np.abs(qp-q).max()
    res.append((c1+c2,c1,c2,a,bx,by))
res.sort()
for r in res[:10]: print(np.round(r,3))
