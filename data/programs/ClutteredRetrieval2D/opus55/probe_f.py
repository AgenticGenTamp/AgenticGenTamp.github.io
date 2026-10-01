from probe_lib import *
from geom import *
import itertools
rng=np.random.default_rng(0)
def objs(o):
    out=[]
    for n in names(o):
        if n.startswith('obstruction') or n=='target_block':
            out.append((n,[g(o,n,f) for f in ['x','y','theta','width','height']]))
    return out
def parts(x,y,th,aj,gmode,armw):
    u=np.array([np.cos(th),np.sin(th)]); p=np.array([x,y])
    gc=p+u*(aj+(0.005 if gmode=='front' else 0))
    parts=[('circle',p,0.1),('poly',rect_centered(*gc,th,0.01,0.07))]
    if armw>0: ac=p+u*aj/2; parts.append(('poly',rect_centered(*ac,th,aj,armw)))
    return parts
def collide(x,y,th,aj,ob,mode,gmode,armw,wall=True):
    ps=parts(x,y,th,aj,gmode,armw)
    walls=[np.array([[-1,-1],[0,-1],[0,3.5],[-1,3.5]]),np.array([[2.5,-1],[3.5,-1],[3.5,3.5],[2.5,3.5]]),
           np.array([[-1,-1],[3.5,-1],[3.5,0],[-1,0]]),np.array([[-1,2.5],[3.5,2.5],[3.5,3.5],[-1,3.5]])]
    polys=[rect_corners(*b,mode=mode) for n,b in ob]+walls
    for P in polys:
        for kind,*a in ps:
            if kind=='circle' and circ_poly(a[0],a[1],P): return True
            if kind=='poly' and poly_poly(a[0],P): return True
    return False
hyps=list(itertools.product(['corner','center'],['center','front'],[0,0.01]))
score={h:[0,0] for h in hyps}; pushes=0; partial=0
for sd in range(12):
    o,_=env.reset(seed=sd); ob=objs(o)
    bx,by=g(o,'target_block','x'),g(o,'target_block','y')
    for k in range(150):
        r=rob(o)
        if rng.random()<0.6:
            d=np.array([bx-r[0],by-r[1]]); d=d/np.linalg.norm(d)*0.05
            a=A(d[0],d[1],rng.uniform(-.196,.196),rng.uniform(-.1,.1))
        else: a=A(*rng.uniform([-.05,-.05,-.196,-.1],[.05,.05,.196,.1]))
        o2,*_=env.step(a); r2=rob(o2)
        if objs(o2)!=ob: pushes+=1; ob=objs(o2)
        rej = r2[:4]==r[:4]
        if not rej and (abs(r2[0]-r[0]-a[0])>1e-3 or abs(r2[1]-r[1]-a[1])>1e-3): partial+=1
        nth=r[2]+a[2]; naj=np.clip(r[3]+a[3],0.1,0.2)
        for h in hyps:
            pred=collide(r[0]+a[0],r[1]+a[1],nth,naj,ob,*h)
            score[h][0 if pred==rej else 1]+=1
        o=o2
print('pushes',pushes,'partial',partial)
for h in hyps: print(h,score[h])
