import numpy as np
from helper import Sim
rows=[]
for seed in range(20):
    s=Sim(seed); o=s.obs.copy(); s.close()
    C=o[9:11]; th=o[11]; a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    L=C+a*o[18]; S=C+b*o[19]
    rows.append([seed,o[0],o[1],o[2],o[9],o[10],o[11],o[20],o[21],o[29],o[30],
                 np.linalg.norm(o[20:22]-o[29:31]),o[17],o[18],o[19],o[28],o[37],o[3],o[5],o[7],o[8],
                 L[0],L[1],S[0],S[1]])
A=np.array(rows)
names=['seed','rx','ry','rth','hx','hy','hth','mx','my','tx','ty','|M-T|','hw','hl1','hl2','mr','tr','brad','armlen','gh','gw','Lx','Ly','Sx','Sy']
print("seed rx ry hx hy hth mx my tx ty |M-T|")
for r in A: print(" ".join(f"{v:7.3f}" for v in r[[0,1,2,4,5,6,7,8,9,10,11]]))
print("\ncol  min      max")
for i,n in enumerate(names):
    print(f"{n:7s} {A[:,i].min():8.4f} {A[:,i].max():8.4f}")
# angle of M-T
d=A[:,7:9]-A[:,9:11]
print("\nM-T angle deg:", np.round(np.degrees(np.arctan2(d[:,1],d[:,0])),1))
print("min my seeds:", A[np.argsort(A[:,8])[:5],0].astype(int), np.round(np.sort(A[:,8])[:5],3))
