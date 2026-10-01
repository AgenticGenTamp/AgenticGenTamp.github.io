import numpy as np, json
from fk import fk
d=json.load(open('calib.json')); base=np.array(d['base'])
def quat2R(qx,qy,qz,qw):
    n=np.array([qx,qy,qz,qw]); n=n/np.linalg.norm(n); x,y,z,w=n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
rows=d['rows']
Rrels=[];offs=[]
for r in rows:
    q=np.array(r['q']); M=fk(q)
    pp=r['part']; Pp=np.array(pp[:3])-np.array([base[0],base[1],0.0]); Rp=quat2R(*pp[3:])
    Rrel=M[:3,:3].T@Rp
    off=M[:3,:3].T@(Pp-M[:3,3])
    Rrels.append(Rrel); offs.append(off)
Rrels=np.array(Rrels); offs=np.array(offs)
print("offset in tool frame: mean",np.round(offs.mean(0),4)," std",np.round(offs.std(0),4))
print("Rrel std per elem:\n",np.round(Rrels.std(0),4))
print("Rrel mean:\n",np.round(Rrels.mean(0),4))
# also world-frame offset
offw=[]
for r in rows:
    q=np.array(r['q']); M=fk(q); pp=r['part']
    Pp=np.array(pp[:3])-np.array([base[0],base[1],0.0])
    offw.append(Pp-M[:3,3])
offw=np.array(offw); print("offset world: mean",np.round(offw.mean(0),4),"std",np.round(offw.std(0),4))

from ik import fk_full
print("\n--- error expressed in each joint frame ---")
errs={i:[] for i in range(7)}
for r in rows:
    q=np.array(r['q']); M,Ms=fk_full(q); pp=r['part']
    Pp=np.array(pp[:3])-np.array([base[0],base[1],0.0])
    e=Pp-M[:3,3]
    for i in range(7):
        errs[i].append(Ms[i][:3,:3].T@e)
for i in range(7):
    a=np.array(errs[i]); print(i,"mean",np.round(a.mean(0),4),"std",np.round(a.std(0),4))
