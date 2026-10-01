from lib import *
import sys
seed=int(sys.argv[1]); side=sys.argv[2]; stp=float(sys.argv[3]); dxo=float(sys.argv[4]) if len(sys.argv)>4 else -0.3
e=E(seed)
e.grasp_hook(1.2)
th=np.pi/2
x,y,rth=e.robot_for_hook(e.obs[9],e.obs[10],th)
e.goto(e.obs[0],e.obs[1],rth,vac=1)
bx,by=e.obs[20:22]
sgn=1 if side=='A' else -1   # A: start above moving down
vx=bx+dxo; vy=by+sgn*0.2
x,y,rth=e.robot_for_hook(vx,vy,th)
# approach: go far left first then over
x0,y0,_=e.robot_for_hook(bx-0.5,e.obs[10],th)
e.goto(x0,e.obs[1],rth,vac=1); e.goto(x0,y,rth,vac=1); e.goto(x,y,rth,vac=1)
print('start hook',e.obs[9:12],'btn',e.obs[20:22])
k=0
for i in range(200):
    pb=e.obs[20:22].copy(); ph=e.obs[9:11].copy()
    te=e.st([0,-sgn*stp,0,0,1])
    if not np.allclose(pb,e.obs[20:22]):
        print(f'step{i} vy-by_after {e.obs[10]-pb[1]:.4f} d {e.obs[20:22]-pb} ',te); k+=1
        if k>=4: break
    if np.allclose(ph,e.obs[9:11]): print('blocked',i); break
