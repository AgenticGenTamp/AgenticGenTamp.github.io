import numpy as np
from scipy.spatial.transform import Rotation

# Candidate Kinova link geometry; calibrated against grasp observations.
LINKS = [(0,0,.15643,np.pi), (0,.005375,-.12838,np.pi/2),
         (0,-.21038,-.006375,-np.pi/2),(0,.006375,-.21038,np.pi/2),
         (0,-.20843,-.006375,-np.pi/2),(0,0,-.10593,np.pi/2),
         (0,-.10593,0,-np.pi/2)]

def fk(q):
    T=np.eye(4)
    for a,(x,y,z,r) in zip(q,LINKS):
        A=np.eye(4)
        A[:3,3]=[x,y,z]
        A[:3,:3]=Rotation.from_euler('x',r).as_matrix() @ Rotation.from_euler('z',a).as_matrix()
        T=T@A
    return T

if __name__=='__main__':
    q=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2])
    for change in [-.6,-.4,-.2,0,.2,.4,.6]:
        p=q.copy();p[1]+=change;p[5]-=change
        T=fk(p)
        print(change,np.round(T[:3,3],4),np.round(T[:3,:3] @ [0,0,-.15],4))
