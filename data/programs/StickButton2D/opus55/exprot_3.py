from exprot import *
from exprot_2 import corners, bb, acc
seed=0
obs,acts=to_grasp(seed); o=obs
A=lambda dx=0,dy=0,dth=0: np.array([dx,dy,dth,0,1.0],dtype=np.float32)
P=[]
def mv(o,dx=0,dy=0,dth=0,tag=''):
    o2=step(o,dx,dy,dth); a=acc(o,o2); P.append(A(dx,dy,dth))
    if not a or tag: print('  ',tag,'dx %.4f dy %.4f dth %.4f'%(dx,dy,dth),'ACCEPT' if a else 'REJECT', bb(rd(o2)[1]))
    return o2,a
def branch(extra,tag):
    o=replay(seed,acts+P)
    for e in extra:
        o2,*_=env.step(A(*e)); a=acc(o,o2); o=o2
        print('   branch',tag,e,'ACCEPT' if a else 'REJECT',bb(rd(o)[1]))
for i in range(26): o,_=mv(o,dx=0.05)
r,s,_=rd(o); px=r['x']; rel=[(cx-px,cy-r['y']) for cx,cy in corners(s)]
def f(phi): c,sn=math.cos(phi),math.sin(phi); return min(c*u-sn*v for u,v in rel)
phis=np.linspace(0,math.pi,200001); fv=[f(p) for p in phis]; k=int(np.argmin(fv)); ph=phis[k]; fs=fv[k]
done=0
while ph-done>1e-9:
    d=min(0.196,ph-done); o,a=mv(o,dth=d); done+=d
# approach wall in steps to find threshold
shift=r['x']+fs-0.01
while shift>1e-9:
    d=min(0.05,shift); o,a=mv(o,dx=-d); shift-=d
while True:
    o2,a=mv(o,dx=-0.0005)
    if not a: P.pop(); break
    o=o2
r,s,_=rd(o); print('threshold: last ok leftmost',bb(s),' (next -0.0005 rejected)')
L=r['x']+fs
# now: rotate back -0.098, move left 0.002 more (mid leftmost ~ L-0.002, ends ~ L-0.002+0.0063)
o,_=mv(o,dth=-0.098,tag='back'); o,_=mv(o,dx=-0.002,tag='left .002 (should be ok, stick not at extremum)')
r,s,_=rd(o); print('pred mid leftmost',r['x']+fs,'pred end',r['x']+f(ph+0.098))
branch([(0,0,0.196)],'sweep through mid')
branch([(0,0,0.098)],'to mid')
branch([(0.01,0,0),(0,0,0.196),(-0.01,0,0)],'end pose via detour')
branch([(0.002,0,0.196)],'combined dx+.002,dth .196')
