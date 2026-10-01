from probe_lib import *
import sys
seed=int(sys.argv[2]); pr=P(seed); obj=16
yaw=np.arctan2(2*(pr.obs[19]*pr.obs[22]),1-2*pr.obs[22]**2)
psi=yaw+float(sys.argv[1])
c=pr.obs[obj:obj+3].copy(); hz=pr.obs[obj+15]/2
tipz=c[2]-hz+float(sys.argv[4])
pr.grip=0
pr.goto(pr.ik([c[0],c[1],c[2]+hz+0.1],psi))
p,why,qe=pr.line([c[0],c[1],c[2]+hz+0.1],[c[0],c[1],tipz],psi,ds=0.003,watch=obj,thr=0.003)
print('yaw',yaw,'descend',why,'obj moved',pr.obs[obj:obj+3]-c)
pr.grip=1
for i in range(int(sys.argv[3])): pr.step(np.r_[np.zeros(10),1])
print('after close obj moved',pr.obs[obj:obj+3]-c)
pr.goto(pr.ik([c[0],c[1],c[2]+0.2],psi))
print('lift: obj',pr.obs[obj:obj+7],'rel',pr.obs[obj:obj+3]-pr.fk())
