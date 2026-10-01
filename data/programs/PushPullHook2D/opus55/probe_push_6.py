from lib import *
import sys
seed=int(sys.argv[1]); deltas=[float(v) for v in sys.argv[2].split(',')]
th=np.pi/2
e=E(seed)
tg=e.obs[29:31].copy()
e.grasp_hook(1.2)
x,y,rth=e.robot_for_hook(e.obs[9],e.obs[10],th)
e.goto(e.obs[0],e.obs[1],rth,vac=1)
bx,by=e.obs[20:22]
def hg(vx,vy):
    x,y,rth=e.robot_for_hook(vx,vy,th); return e.goto(x,y,rth,vac=1,verbose=False)
x0,y0,_=e.robot_for_hook(bx-0.6,by-0.2,th)
e.goto(x0,e.obs[1],rth,vac=1); e.goto(x0,y0,rth,vac=1); hg(bx-0.3,by-0.2)
print('btn',e.obs[20:22],'tgt',tg,'dist',np.linalg.norm(e.obs[20:22]-tg))
for dl in deltas:
    bx,by=e.obs[20:22]
    hg(e.obs[9], by-0.06)
    assert np.allclose(e.obs[20:22],[bx,by])
    te=e.st([0,0.06-dl,0,0,1]); d1=e.obs[21]-by
    te=e.st([0,0,0,0,1]) or te
    print('delta',dl,'d1',round(d1,4),'total',round(e.obs[21]-by,4),'dist',round(float(np.linalg.norm(e.obs[20:22]-tg)),4),te)
    if te: break
    hg(e.obs[9], e.obs[10]-0.02)
