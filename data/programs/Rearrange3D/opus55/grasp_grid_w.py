from probe_lib import *
import sys
dpsi=float(sys.argv[1]); seed=int(sys.argv[2]); obj=int(sys.argv[3]); h=float(sys.argv[4])
pr=P(seed)
for i in range(int(sys.argv[6]) if len(sys.argv)>6 else 0): pr.step(np.zeros(11))
o=pr.obs
yaw=np.arctan2(2*(o[obj+3]*o[obj+6]+o[obj+4]*o[obj+5]),1-2*(o[obj+5]**2+o[obj+6]**2))
psi=yaw+dpsi
c=o[obj:obj+3].copy(); hz=o[obj+15]/2
tipz=c[2]-hz+h
pr.grip=0
pr.goto(pr.ik([c[0],c[1],c[2]+hz+0.1],psi))
p,why,qe=pr.line([c[0],c[1],c[2]+hz+0.1],[c[0],c[1],tipz],psi,ds=0.004,watch=obj,thr=0.004)
m1=np.linalg.norm(pr.obs[obj:obj+2]-c[:2])
NC=int(sys.argv[5]) if len(sys.argv)>5 else 12
for i in range(NC): pr.step(np.r_[np.zeros(10),1])
m2=np.linalg.norm(pr.obs[obj:obj+2]-c[:2])
pr.grip=1; pr.goto(pr.ik([c[0],c[1],c[2]+0.2],psi))
rel=pr.obs[obj:obj+3]-pr.fk()
print(f"obj{obj} dpsi{dpsi:.2f} h{h:+.3f} desc:{why} mv{m1:.3f} close_mv{m2:.3f} lifted {pr.obs[obj+2]-c[2]:.3f} qw {pr.obs[obj+3]:.3f} rel {rel.round(3)}")
