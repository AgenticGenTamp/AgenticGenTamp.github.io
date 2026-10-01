from probe_lib import *
import sys
dpsi=float(sys.argv[1]); seed=int(sys.argv[2]); obj=16; tipz=float(sys.argv[3]); mode=sys.argv[4]
pr=P(seed)
for i in range(60): pr.step(np.zeros(11))
o=pr.obs
yaw=np.arctan2(2*(o[obj+3]*o[obj+6]+o[obj+4]*o[obj+5]),1-2*(o[obj+5]**2+o[obj+6]**2))
psi=yaw+dpsi
c=o[obj:obj+3].copy(); b=o[0:2]
pr.grip=0
pr.goto(pr.ik([c[0],c[1],0.6],psi))
p,why,qe=pr.line([c[0],c[1],0.6],[c[0],c[1],tipz],psi,ds=0.004,watch=obj,thr=0.004)
for i in range(4): pr.step(np.r_[np.zeros(10),1])
pr.grip=1
if mode=='lift':
    pr.goto(pr.ik([c[0],c[1],tipz+0.1],psi))
else:
    d=b-c[:2]; tgt=c[:2]+d*(1-0.11/np.linalg.norm(d))
    p,why2,qe=pr.line([c[0],c[1],tipz],[tgt[0],tgt[1],tipz+0.003],psi,ds=0.004)
print(f"dpsi{dpsi:.2f} tipz{tipz} {mode} desc:{why} obj {pr.obs[obj:obj+3].round(3)} from {c.round(3)} qw {pr.obs[obj+3]:.3f} tip {pr.fk().round(3)} dbowl {np.linalg.norm(pr.obs[obj:obj+2]-pr.obs[0:2]):.3f}")
