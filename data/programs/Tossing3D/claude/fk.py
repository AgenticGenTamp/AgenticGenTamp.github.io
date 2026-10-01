import numpy as np
def rpy(r,p,y):
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])
def H(xyz,r,p,y):
    M=np.eye(4); M[:3,:3]=rpy(r,p,y); M[:3,3]=xyz; return M
PI=np.pi
JOINTS=[((0,0,0.15643),(PI,0,0)),
        ((0,0.005375,-0.12838),(PI/2,0,0)),
        ((0,-0.21038,-0.006375),(-PI/2,0,0)),
        ((0,0.006375,-0.21038),(PI/2,0,0)),
        ((0,-0.20843,-0.006375),(-PI/2,0,0)),
        ((0,0.00017505,-0.10593),(PI/2,0,0)),
        ((0,-0.10593,-0.00017505),(-PI/2,0,0))]
EE=H((0,0,-0.0615),PI,0,0)
def Rz(t):
    M=np.eye(4); c,s=np.cos(t),np.sin(t); M[0,0]=c;M[0,1]=-s;M[1,0]=s;M[1,1]=c; return M
def fk(q,tool=0.0):
    M=np.eye(4)
    for i,(xyz,r) in enumerate(JOINTS):
        M=M@H(xyz,*r)@Rz(q[i])
    M=M@EE
    if tool: M=M@H((0,0,tool),0,0,0)
    return M
if __name__=="__main__":
    print("zeros",np.round(fk(np.zeros(7))[:3,3],4))
    q=np.array([0,-0.349,3.1416,-2.548,0,-0.873,1.5708])
    print("init",np.round(fk(q),4))
