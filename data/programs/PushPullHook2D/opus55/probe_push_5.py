from lib import *
import sys
seed=int(sys.argv[1]); mode=sys.argv[2]
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
print('btn',e.obs[20:22],'tgt',tg)
if mode=='thr':
    # fine pushes in +y by delta using short arm from below
    for k in range(60):
        bx,by=e.obs[20:22]
        hg(e.obs[9], by-0.05-0.001)
        if not np.allclose(e.obs[20:22],[bx,by]): print('moved during approach'); 
        bx,by=e.obs[20:22]
        te=e.st([0,0.051-0.004,0,0,1])
        d=e.obs[20:22]-[bx,by]
        print(k,'d',d,'dist',round(float(np.linalg.norm(e.obs[20:22]-tg)),4),te)
        if te: break
        hg(e.obs[9], e.obs[10]-0.02)
elif mode=='zero':
    bx,by=e.obs[20:22]
    hg(e.obs[9], by-0.05-0.001)
    for s in [0.04,0,0,0.001,0.001,-0.001,0]:
        p=e.obs[20:22].copy(); e.st([0,s,0,0,1]); print('s',s,'d',e.obs[20:22]-p,'rel',e.obs[10]-e.obs[21])
elif mode=='wall':
    for k in range(200):
        p=e.obs[20:22].copy(); ph=e.obs[9:11].copy(); e.st([0,0.01,0,0,1])
        if np.allclose(ph,e.obs[9:11]): print('hook blocked', e.obs[0:2],e.obs[9:11],'btn',e.obs[20:22]); break
    print('end btn',e.obs[20:22],'hook',e.obs[9:11],'robot',e.obs[:2])
