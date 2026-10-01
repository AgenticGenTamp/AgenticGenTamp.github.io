import numpy as np, approach as A
rng=np.random.default_rng(0)
bad=0
for t in range(2000):
    N=rng.integers(1,12)
    polys=[]
    for i in range(N):
        x,y=rng.uniform(0,2.5,2); w,h=rng.uniform(0.02,0.4,2); th=rng.uniform(-3,3)
        c,s=np.cos(th),np.sin(th); L=np.array([[0,0],[w,0],[w,h],[0,h]])
        polys.append(np.array([x,y])+L@np.array([[c,s],[-s,c]]))
    import inspect
    ob=A.Obstacles(np.array(polys)) if 'Obstacles' in dir(A) else None
    M=50; Q=[]
    for i in range(M):
        x,y=rng.uniform(0,2.5,2); w,h=rng.uniform(0.005,0.2,2); th=rng.uniform(-3,3)
        c,s=np.cos(th),np.sin(th); L=np.array([[0,0],[w,0],[w,h],[0,h]])
        Q.append(np.array([x,y])+L@np.array([[c,s],[-s,c]]))
    Q=np.array(Q)
    for m in [0,0.01,0.05]:
        a,b=ob.poly_hit(Q,m),ob.poly_hit_full(Q,m); bad+=(a!=b).sum()
        if (a!=b).any(): print(m,a[a!=b],b[a!=b])
print('mismatches',bad)
