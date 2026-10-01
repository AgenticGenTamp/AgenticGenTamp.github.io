import numpy as np
# Kinova Gen3 classic DH (from user guide)
D = [-(0.1564+0.1284), -(0.0054+0.0064), -(0.2104+0.2104), -(0.0064+0.0064), -(0.2084+0.1059), 0.0, -(0.1059+0.0615)]
ALPHA = [np.pi/2]*6 + [np.pi]
OFF = [0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
def dh(a, alpha, d, th):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(th), np.sin(th)
    return np.array([[ct, -st*ca, st*sa, a*ct],[st, ct*ca, -ct*sa, a*st],[0, sa, ca, d],[0,0,0,1]])
def fk_arm(q, tool=0.0):
    T = np.diag([1.,-1.,-1.,1.])  # base frame: rot pi about x
    frames=[T.copy()]
    for i in range(7):
        T = T @ dh(0, ALPHA[i], D[i], q[i]+OFF[i]); frames.append(T.copy())
    Tt = np.eye(4); Tt[2,3]=tool
    return T @ Tt, frames
if __name__=='__main__':
    q=[0,-0.35,-np.pi,-2.5,0,-0.87,np.pi/2]
    T,fr=fk_arm(q,0.12)
    np.set_printoptions(precision=3,suppress=True)
    print(T)
    for f in fr: print(f[:3,3])
    print(fk_arm([0]*7)[0][:3,3])
