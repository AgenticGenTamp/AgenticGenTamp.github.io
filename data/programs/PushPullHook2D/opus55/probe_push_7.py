from lib import *
import sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
side=sys.argv[2] if len(sys.argv)>2 else 'L'
e=E(seed)
print('init robot',e.obs[:3],'hook',e.obs[9:12],'btn',e.obs[20:22],'tgt',e.obs[29:31])
e.grasp_hook(1.2)
th=np.pi/2
x,y,rth=e.robot_for_hook(e.obs[9],e.obs[10],th)
e.goto(e.obs[0],e.obs[1],rth,vac=1)
bx,by=e.obs[20:22]
print('after rot hook',e.obs[9:12],'btn',bx,by)
sgn=-1 if side=='L' else 1
vx=bx+sgn*0.2; vy=by+0.3
x,y,rth=e.robot_for_hook(vx,vy,th)
e.goto(x,e.obs[1],rth,vac=1); e.goto(x,y,rth,vac=1)
b0=e.obs[20:22].copy()
print('start hook',e.obs[9:12],'btn',b0)
for i in range(120):
    pb=e.obs[20:22].copy(); ph=e.obs[9:11].copy()
    te=e.st([-sgn*0.01,0,0,0,1])
    if i%10==0 or not np.allclose(pb,e.obs[20:22]) and False:
        print(f'step{i} hook_before {ph} hook {e.obs[9:11]} btn {pb}->{e.obs[20:22]} gap_before {sgn*(ph[0]-pb[0]):.4f} gap_after {sgn*(e.obs[9]-pb[0]):.4f} d {e.obs[20:22]-pb}',te)
    if np.allclose(ph,e.obs[9:11]): print('blocked',i); break
