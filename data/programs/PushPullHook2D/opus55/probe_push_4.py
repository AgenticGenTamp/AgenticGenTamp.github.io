from lib import *
import sys
# generic: place hook vertex at btn + (ox,oy) with theta th, then step action a repeatedly
seed=int(sys.argv[1]); ox=float(sys.argv[2]); oy=float(sys.argv[3]); ax=float(sys.argv[4]); ay=float(sys.argv[5]); ath=float(sys.argv[6]) if len(sys.argv)>6 else 0
th=float(sys.argv[7]) if len(sys.argv)>7 else np.pi/2
N=int(sys.argv[8]) if len(sys.argv)>8 else 4
e=E(seed)
print('tgt',e.obs[29:31],'r',e.obs[28],e.obs[37])
e.grasp_hook(1.2)
x,y,rth=e.robot_for_hook(e.obs[9],e.obs[10],th)
e.goto(e.obs[0],e.obs[1],rth,vac=1)
bx,by=e.obs[20:22]
x,y,rth=e.robot_for_hook(bx+ox,by+oy,th)
x0,y0,_=e.robot_for_hook(bx-0.6,by+oy,th)
e.goto(x0,e.obs[1],rth,vac=1); e.goto(x0,y,rth,vac=1); e.goto(x,y,rth,vac=1)
print('start hook',e.obs[9:12],'btn',e.obs[20:22], 'rel',e.obs[9:11]-e.obs[20:22])
k=0
for i in range(300):
    pb=e.obs[20:22].copy(); ph=e.obs[9:12].copy()
    te=e.st([ax,ay,ath,0,1])
    if not np.allclose(pb,e.obs[20:22]):
        d=e.obs[20:22]-pb
        print(f'step{i} rel_after {e.obs[9:11]-pb} hth {e.obs[11]:.3f} d {d} |d| {np.linalg.norm(d):.4f} ang {np.degrees(np.arctan2(d[1],d[0])):.1f} tgtdist {np.linalg.norm(e.obs[20:22]-e.obs[29:31]):.4f}',te); k+=1
        if k>=N: break
    if np.allclose(ph,e.obs[9:12]): print('blocked',i, e.obs[9:12]); break
