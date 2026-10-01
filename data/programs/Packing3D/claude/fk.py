import numpy as np

def rpy(r,p,y):
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp,   cp*sr,          cp*cr]])
def T(xyz, R):
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=xyz; return M
def rotz(q):
    c,s=np.cos(q),np.sin(q); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# gen3 7dof urdf guess
JOINTS=[((0,0,0.15643),(-np.pi,0,0)),
        ((0,0.005375,-0.12838),(np.pi/2,0,0)),
        ((0,-0.21038,-0.006375),(-np.pi/2,0,0)),
        ((0,0.006375,-0.21038),(np.pi/2,0,0)),
        ((0,-0.20843,-0.006375),(-np.pi/2,0,0)),
        ((0,0.00017505,-0.10593),(np.pi/2,0,0)),
        ((0,-0.10593,-0.00017505),(-np.pi/2,0,0))]
TOOL=((0,0,-0.0615),(np.pi,0,0))

def fk(q, tool_z=0.0):
    M=np.eye(4)
    for i,(xyz,r) in enumerate(JOINTS):
        M = M @ T(np.array(xyz), rpy(*r)) @ T(np.zeros(3), rotz(q[i]))
    M = M @ T(np.array(TOOL[0]), rpy(*TOOL[1]))
    if tool_z: M = M @ T(np.array([0,0,tool_z]), np.eye(3))
    return M

if __name__=="__main__":
    q=[0.0,-0.35,-3.1416,-2.5,0.0,-0.87,1.5708]
    M=fk(q)
    print("home ee pos", M[:3,3])
    print("R\n", M[:3,:3])
    q0=[0,0,0,0,0,0,0]
    print("zero ee", fk(q0)[:3,3])
