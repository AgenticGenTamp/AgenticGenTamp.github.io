import numpy as np, fk
D=np.load("calib.npy")
gtf_p=np.array([0.05791,0.01004,0.00007]); gtf_q=np.array([-0.5,-0.49999,0.5,0.50001])
def q2R(q):
    x,y,z,w=q; n=np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
Rg=q2R(gtf_q)
tools=[]
for row in D:
    base=row[:3]; q=row[3:10]; bp=row[10:13]; bq=row[13:17]
    Rb=q2R(bq)
    Rt = Rb @ Rg.T
    pt = bp - Rt @ gtf_p
    tools.append((base,q,pt,Rt))
# compare with fk (base frame)
errs=[]
for base,q,pt,Rt in tools:
    p,Rm,_=fk.fk_arm(q,0.0)
    c,s=np.cos(base[2]),np.sin(base[2]); Rbz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    pw = Rbz@p + np.array([base[0],base[1],0])
    Rw = Rbz@Rm
    # error expressed in tool frame
    e_local = Rw.T @ (pt-pw)
    errs.append((pt-pw, e_local, np.round(Rw.T@Rt,3)))
E=np.array([e[0] for e in errs]); EL=np.array([e[1] for e in errs])
print("world err mean",np.round(E.mean(0),5),"std",np.round(E.std(0),5))
print("local err mean",np.round(EL.mean(0),5),"std",np.round(EL.std(0),5))
print("rot rel sample",errs[0][2], errs[10][2])
