import numpy as np
def rotx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rotz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def T(R,p):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=p; return M
PI=np.pi
# (xyz, rx) per joint from Kinova Gen3 7DOF URDF
LINKS=[((0,0,0.15643),-PI),
       ((0,0.005375,-0.12838),PI/2),
       ((0,-0.21038,-0.006375),-PI/2),
       ((0,0.006375,-0.21038),PI/2),
       ((0,-0.20843,-0.006375),-PI/2),
       ((0,0.00017505,-0.10593),PI/2),
       ((0,-0.10593,-0.00017505),-PI/2)]
TOOL=((0,0,-0.0615),PI)
def fk(q, tool_z=0.0):
    M=np.eye(4)
    for i,(xyz,rx) in enumerate(LINKS):
        M=M@T(rotx(rx),np.array(xyz))@T(rotz(q[i]),np.zeros(3))
    M=M@T(rotx(TOOL[1]),np.array(TOOL[0]))
    M=M@T(np.eye(3),np.array([0,0,tool_z]))
    return M
if __name__=="__main__":
    q0=[0.0,-0.3491,3.1416,-2.5482,-0.0,-0.8727,1.5708]
    M=fk(q0)
    np.set_printoptions(precision=4,suppress=True)
    print("EE in armbase frame:",M[:3,3])
    print("R\n",M[:3,:3])
    print("zeros:",fk([0]*7)[:3,3])
    print("home:",fk([0,0.262,3.1416,-2.269,0,0.96,1.5708])[:3,3])
