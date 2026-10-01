"""Front depth scan: rod pointing +x pushed into cupboard at (y,z); prints tip x where blocked."""
import sys, json; sys.path.insert(0,'/sandbox/scratch')
from plib import *
seed=int(sys.argv[1]); k=None if sys.argv[2]=='n' else int(sys.argv[2])
ys=[float(v) for v in sys.argv[3].split(',')]; zs=[float(v) for v in sys.argv[4].split(',')]
BX=float(sys.argv[5]) if len(sys.argv)>5 else 1.3
p=P(seed,k); S=p.S; R=Rdown(np.pi/2)
rods=sorted(S.rods(), key=lambda n: np.linalg.norm(S.rods()[n][:2]-S.base()[:2]))
p.pick(rods[0])
g=grasp_point_world(S)
moveto_slow(p,[g[0],g[1],0.4],R)
print('reoriented',p.rel()[1].round(2),len(S.rew),flush=True)
out={}
for yc in ys:
    moveto_slow(p,[BX+0.45,yc,0.35],R,bt=[BX,yc,0])
    for z in zs:
        d,ax=p.rel()
        if np.linalg.norm(d)>0.03 or abs(ax[0])<0.99: print('ROD LOST',yc,z,d.round(3),ax.round(2),flush=True); sys.exit()
        r=p.push(yc,z,BX+0.45,BX+0.9,R,dx=0.005,sub=10,thr=0.006)
        tip=None if r is None else round(r[0]-0.01+0.15+d[0],3)
        out[(yc,z)]=tip
        print('Y %.3f Z %.3f tip %s raw %s ax %s steps %d'%(yc,z,tip,r,p.rel()[1].round(3),len(S.rew)),flush=True)
print('nonm1',LOG)
