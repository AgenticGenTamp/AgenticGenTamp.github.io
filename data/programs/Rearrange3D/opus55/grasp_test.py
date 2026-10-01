from probe_lib import *
import sys
seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
pr=P(seed); psi=float(sys.argv[1]); obj=int(sys.argv[2])
c=pr.obs[obj:obj+3].copy(); hz=pr.obs[obj+15]/2
tipz=c[2]-hz+0.025
pr.grip=0
pr.goto(pr.ik([c[0],c[1],c[2]+hz+0.1],psi))
p,why,qe=pr.line([c[0],c[1],c[2]+hz+0.1],[c[0],c[1],tipz],psi,ds=0.003,watch=obj,thr=0.003)
print('descend',why,'obj moved',pr.obs[obj:obj+3]-c)
for i in range(20): pr.grip=min(1,0.1*(i+1)); pr.step(np.r_[np.zeros(10),pr.grip])
print('after close obj moved',pr.obs[obj:obj+3]-c)
pr.goto(pr.ik([c[0],c[1],c[2]+0.2],psi))
print('lift: obj',pr.obs[obj:obj+7],'fk',pr.fk(),'rel',pr.obs[obj:obj+3]-pr.fk())
