import numpy as np, fk
D=np.load("calib.npy")
gtf_p=np.array([0.05791,0.01004,0.00007]); gtf_q=np.array([-0.5,-0.49999,0.5,0.50001])
def q2R(q):
    x,y,z,w=q; n=np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
Rg=q2R(gtf_q)
A=[];B=[]
for row in D:
    base=row[:3]; q=row[3:10]; bp=row[10:13]; bq=row[13:17]
    Rt = q2R(bq) @ Rg.T
    pt = bp - Rt @ gtf_p   # true tool world pos
    p,Rm,frames=fk.fk_arm(q,0.0)
    c,s=np.cos(base[2]),np.sin(base[2]); Rbz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pw=Rbz@p+np.array([base[0],base[1],0])
    # jacobian wrt link translation offsets: delta applied in frame BEFORE joint i rotation
    # p = torso + sum_i R_{i-1} t_i (+ R_7 TOOL), where R_{i-1} is rotation before joint i
    cols=[]
    Rprev=np.eye(3)
    cols.append(Rbz)  # torso offset (base frame)
    for i in range(7):
        cols.append(Rbz@Rprev)
        Rprev = Rprev @ fk.rot(fk.LINKS[i][1], q[i])
    cols.append(Rbz@Rprev)  # tool offset
    A.append(np.hstack(cols)); B.append(pt-pw)
A=np.vstack(A); B=np.concatenate(B)
n=A.shape[1]
lam=1e-3
x=np.linalg.solve(A.T@A+lam*np.eye(n), A.T@B)
res=A@x-B
print("params (torso, l1..l7, tool):")
names=["torso"]+["l%d"%i for i in range(1,8)]+["tool"]
for i,nm in enumerate(names): print(" ",nm,np.round(x[3*i:3*i+3],4))
print("residual max",np.abs(res).max(),"rms",np.sqrt((res**2).mean()))
