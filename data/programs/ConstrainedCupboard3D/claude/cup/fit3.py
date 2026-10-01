import sys; sys.path.insert(0,"/sandbox")
import numpy as np
from scipy.optimize import brentq
p=np.load('/sandbox/cup/cam2.npy')
def rod3(rv):
    t=np.linalg.norm(rv); k=rv/t
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(t)*K+(1-np.cos(t))*K@K
def proj(pts):
    C=p[:3];Rm=rod3(p[3:6]);f=p[6]
    pc=(Rm@(np.asarray(pts,float)-C).T).T
    return np.stack([320+f*pc[:,0]/pc[:,2],240+f*pc[:,1]/pc[:,2]],1)
def zof(v,x,y=0.0): return brentq(lambda z: proj([[x,y,z]])[0][1]-v,-1.0,4.0)
def xof(v,z,y=0.0): return brentq(lambda x: proj([[x,y,v*0+z]])[0][1]-v,1.5,3.5)
xf=p[7]; print('front face x = %.3f'%xf)
for name,vf,vb in [('base plate',150.5,128.5),('LOW shelf',133.5,111.5),('MID shelf',111.5,89.5),('HIGH shelf',87.5,70.5),('top plate',67.0,47.0)]:
    z=zof(vf,xf)
    try: xb=xof(vb,z)
    except Exception as e: xb=float('nan')
    print('  %-11s front v=%.1f -> z=%.3f ; back edge v=%.1f at same z -> x=%.3f (depth %.3f)'%(name,vf,z,vb,xb,xb-xf))
print()
print('height check: top plate top surface z=%.3f (front) ; total cupboard height'%zof(67.0,xf))
