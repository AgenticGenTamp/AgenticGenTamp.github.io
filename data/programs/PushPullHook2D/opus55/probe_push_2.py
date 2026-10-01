from lib import *
import sys
seed=int(sys.argv[1]); side=sys.argv[2]; stp=float(sys.argv[3]); off=float(sys.argv[4]) if len(sys.argv)>4 else 0.2
e=E(seed)
e.grasp_hook(1.2)
th=np.pi/2
x,y,rth=e.robot_for_hook(e.obs[9],e.obs[10],th)
e.goto(e.obs[0],e.obs[1],rth,vac=1)
bx,by=e.obs[20:22]
sgn=-1 if side=='L' else 1
vx=bx+sgn*off; vy=by+0.3
x,y,rth=e.robot_for_hook(vx,vy,th)
e.goto(e.obs[0],y,rth,vac=1); e.goto(x,y,rth,vac=1)
print('start hook',e.obs[9:12],'btn',e.obs[20:22])
k=0
for i in range(200):
    pb=e.obs[20:22].copy(); ph=e.obs[9:11].copy()
    te=e.st([-sgn*stp,0,0,0,1])
    if not np.allclose(pb,e.obs[20:22]):
        # overlap: L: rect right edge vx+0.05 vs btn left bx-0.05 ; R: rect left vx vs btn right bx+0.05
        ov = (e.obs[9]+0.05)-(pb[0]-0.05) if side=='L' else (pb[0]+0.05)-e.obs[9]
        print(f'step{i} ov {ov:.4f} d {e.obs[20:22]-pb} newsep {-sgn*(e.obs[20]-e.obs[9]):.4f}',te); k+=1
        if k>=4: break
    if np.allclose(ph,e.obs[9:11]): print('blocked',i); break
