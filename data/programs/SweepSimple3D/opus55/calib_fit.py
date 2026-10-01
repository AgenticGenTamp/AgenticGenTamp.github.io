import numpy as np, json, sys, kin
from scipy.optimize import least_squares
files=sys.argv[1:]
sets=[json.load(open(f)) for f in files]
def rot(r,p):
    cr,sr,cp,sp=np.cos(r),np.sin(r),np.cos(p),np.sin(p)
    return np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
def res(x, full=False):
    mxyz,myaw,mr,mp,tl=x[0:3],x[3],x[4],x[5],x[6]
    out=[]
    for si,S in enumerate(sets):
        o=x[7+4*si:10+4*si]; s=x[10+4*si]
        for i,d in enumerate(S):
            p,R=kin.fk_arm(np.array(d['q']),mount_xyz=np.zeros(3),mount_yaw=0.0,tip_len=tl)
            p=kin._rz(myaw)@rot(mr,mp)@p+mxyz; R=kin._rz(myaw)@rot(mr,mp)@R
            bx,by,br=d['base']; Rb=kin._rz(br)
            pw=np.array([bx,by,0])+Rb@p; Rw=Rb@R
            pc=pw+Rw@(o+np.array([0,0,s*i]))
            out.append(pc-np.array(d['cube'][:3]))
    return np.concatenate(out)
x0=np.concatenate([[0.12,0,0.39,0,0,0,0.145],np.zeros(4*len(sets))])
for fix in [True,False]:
    lb=-np.inf*np.ones_like(x0); ub=np.inf*np.ones_like(x0)
    if fix: lb[4:6]=-1e-9; ub[4:6]=1e-9
    x0c=np.clip(x0,lb+1e-12,ub-1e-12) if fix else x0
    r=least_squares(res,x0c,bounds=(lb,ub))
    e=res(r.x).reshape(-1,3)
    print('fix roll/pitch' if fix else 'free', np.round(r.x,4))
    print(' rms per-axis',np.round(np.sqrt((e**2).mean(0)),4),'max norm',np.round(np.linalg.norm(e,axis=1).max(),4))
