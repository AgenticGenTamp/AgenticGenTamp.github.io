import numpy as np, fk
from scipy.optimize import least_squares
D=np.load("calib_data.npy")
print(D.shape)
gtf_p=np.array([0.03,-0.0,-0.02498]); gtf_q=np.array([0,0,0.03713,0.99931])
def q2R(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
Rg=q2R(gtf_q)
# measured tool pose: T_tool = T_block * inv(T_gtf)
meas_p=[];meas_R=[];Q=[];B=[]
for row in D:
    base=row[:3]; q=row[3:10]; bp=row[10:13]; bq=row[13:17]
    Rb=q2R(bq)
    R_tool=Rb@Rg.T
    p_tool=bp-R_tool@gtf_p
    meas_p.append(p_tool); meas_R.append(R_tool); Q.append(q); B.append(base)
meas_p=np.array(meas_p);Q=np.array(Q);B=np.array(B)
def model(theta, q, base):
    tx,ty,tz,tool=theta
    p=np.array([tx,ty,tz]); R=fk._rz(q[0])
    p=p+R@np.array([0.1,0,0]); R=R@fk._ry(q[1])@fk._rx(q[2])
    p=p+R@np.array([0.4,0,0]); R=R@fk._ry(q[3])@fk._rx(q[4])
    p=p+R@np.array([0.321,0,0]); R=R@fk._ry(q[5])@fk._rx(q[6])
    p=p+R@np.array([tool,0,0])
    Rb=fk._rz(base[2])
    return Rb@p+np.array([base[0],base[1],0.0]), Rb@R
def resid(theta):
    out=[]
    for i in range(len(Q)):
        p,R=model(theta,Q[i],B[i])
        out.append(p-meas_p[i])
        Re=R@meas_R[i].T
        out.append(np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])*0.5)
    return np.concatenate(out)
sol=least_squares(resid,[-0.05,0.188,0.99,0.18])
print("theta",np.round(sol.x,5),"rms",np.sqrt(np.mean(sol.fun**2)))
r=resid(sol.x).reshape(-1,3)
print("max pos err",np.abs(r[0::2]).max(),"max rot err",np.abs(r[1::2]).max())
