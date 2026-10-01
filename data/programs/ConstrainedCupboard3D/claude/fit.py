import json, numpy as np, kin
from scipy.optimize import least_squares
D=json.load(open('calib_data.json'))
def rotz(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1.0]])
def rv2R(v):
    th=np.linalg.norm(v)
    if th<1e-12: return np.eye(3)
    k=v/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
def resid(p):
    mx,my,mz=p[0:3]; gp=p[3:6]; gr=p[6:9]
    Rg=rv2R(gr)
    out=[]
    for d in D:
        q=np.array(d['q']); b=np.array(d['b']); rod=np.array(d['rod'])
        Te=kin.arm_fk(q,0.0)
        Rb=rotz(b[2])
        Rw=Rb@Te[:3,:3]
        pw=np.array([b[0],b[1],0])+Rb@(np.array([mx,my,mz])+Te[:3,3])
        # rod pose = ee * grasp
        pr=pw+Rw@gp
        Rr=Rw@Rg
        Rt=kin.quat_to_mat(rod[3:7])
        out.extend(pr-rod[:3])
        out.extend(kin._log_so3(Rt@Rr.T)*0.3)
    return np.array(out)
best=None
for k in range(30):
    x0=np.concatenate([[0.1,0,0.45],np.random.uniform(-0.2,0.2,3),np.random.uniform(-3,3,3)])
    r=least_squares(resid,x0)
    if best is None or r.cost<best.cost: best=r
p=best.x
print("cost",best.cost,"maxres",np.max(np.abs(best.fun)))
print("mount",np.round(p[:3],4),"grasp_pos",np.round(p[3:6],4),"grasp_rot",np.round(p[6:9],4))
res=best.fun.reshape(len(D),6)
print("pos rms",np.round(np.sqrt((res[:,:3]**2).sum(1)),4))
