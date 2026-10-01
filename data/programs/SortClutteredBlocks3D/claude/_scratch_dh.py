import numpy as np
def Rx(a):
    c,s=np.cos(a),np.sin(a); T=np.eye(4); T[1,1]=c;T[1,2]=-s;T[2,1]=s;T[2,2]=c; return T
def Rz(a):
    c,s=np.cos(a),np.sin(a); T=np.eye(4); T[0,0]=c;T[0,1]=-s;T[1,0]=s;T[1,1]=c; return T
def Tr(x,y,z):
    T=np.eye(4); T[:3,3]=[x,y,z]; return T
URDF=[((0,0,0.15643), np.pi),
      ((0,0.005375,-0.12838), np.pi/2),
      ((0,-0.21038,-0.006375), -np.pi/2),
      ((0,0.006375,-0.21038), np.pi/2),
      ((0,-0.20843,-0.006375), -np.pi/2),
      ((0,0.00017505,-0.10593), np.pi/2),
      ((0,-0.10593,-0.00017505), -np.pi/2)]
TOOLF=(Tr(0,0,-0.0615)@Rx(np.pi))
def fk_urdf(q):
    T=np.eye(4)
    for i,(xyz,rx) in enumerate(URDF):
        T=T@Tr(*xyz)@Rx(rx)@Rz(q[i])
    return T@TOOLF
def dh(al,a,d,th):
    return Rz(th)@Tr(0,0,d)@Tr(a,0,0)@Rx(al)
D=[0.2848,0.0118,0.4208,0.0128,0.3143,0.0,0.1674]
def fk_dh(q,alphas,offs):
    T=Rx(np.pi)
    for i in range(7):
        T=T@dh(alphas[i],0.0,-D[i],q[i]+offs[i])
    return T
q=np.array([0.1,-0.4,0.7,-1.2,0.3,0.9,-0.5])
print("URDF q0:", np.round(fk_urdf(np.zeros(7)),6)[:3,:])
print("URDF q :", np.round(fk_urdf(q),6)[:3,:])
# try variants
best=None
import itertools
for offs7 in [0,np.pi]:
  alphas=[np.pi/2]*6+[np.pi]
  offs=[0]+[np.pi]*6
  T=fk_dh(q,alphas,offs)
  print("variantA:", np.round(T[:3,:],6))
