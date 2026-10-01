from exprot import *
def corners(s,w=0.05,h=1.25):
    x,y,th=s; c,sn=math.cos(th),math.sin(th)
    return [(x+c*u-sn*v, y+sn*u+c*v) for u,v in ((0,0),(w,0),(w,h),(0,h))]
def bb(s):
    P=corners(s); xs=[p[0] for p in P]; ys=[p[1] for p in P]
    return 'X[%.4f,%.4f] Y[%.4f,%.4f]'%(min(xs),max(xs),min(ys),max(ys))
def dist(s,bx,by,w=0.05,h=1.25):
    x,y,th=s; c,sn=math.cos(th),math.sin(th); dx,dy=bx-x,by-y
    u=c*dx+sn*dy; v=-sn*dx+c*dy
    return math.hypot(u-min(max(u,0),w), v-min(max(v,0),h))
def goto(o,tx,ty):
    for _ in range(200):
        r,_,_=rd(o); dx=np.clip(tx-r['x'],-.05,.05); dy=np.clip(ty-r['y'],-.05,.05)
        if abs(dx)<1e-6 and abs(dy)<1e-6: break
        o2=step(o,dx,dy)
        if rd(o2)[0]['x']==r['x'] and rd(o2)[0]['y']==r['y']: print('  goto blocked at',round(r['x'],3),round(r['y'],3),bb(rd(o)[1])); return o2
        o=o2
    return o
seed=0
obs,acts=to_grasp(seed); o=obs
# table test 1: vertical stick, move up
o=goto(o,0.513,2.4); r,s,_=rd(o); print('vertical: robot y',round(r['y'],4),bb(s))
o=replay(seed,acts)
# rotate -pi/2 so stick points right
for i in range(8): o=step(o,dth=-math.pi/16)
r,s,_=rd(o); print('horizontal: robot',round(r['x'],3),round(r['y'],3),'th',round(r['theta'],3),'stick',[round(v,3) for v in s],bb(s))
o=goto(o,0.6,2.4); r,s,_=rd(o); print('horizontal moved up: robot y',round(r['y'],4),bb(s))
o=goto(o,0.6,1.0)
# button test: b0 at (2.113,1.801) r .05
r,s,_=rd(o); P=corners(s); top=max(p[1] for p in P)-r['y']
o=goto(o,1.2,1.801-0.05-0.03-top); r,s,_=rd(o); print('pre-button robot',round(r['x'],3),round(r['y'],3),bb(s))
for i in range(8):
    r,s,b=rd(o); o2=step(o,dth=0.02); r2,s2,b2=rd(o2)
    acc=abs(r2['theta']-r['theta'])>1e-7
    bb0=[x for x in b2 if x[0]=='button0'][0]
    print('  rot +0.02',('ACC' if acc else 'REJ'),'dist-to-b0 %.4f (overlap if <0.05)'%dist(s2,2.113,1.801),'b0 color r,g',round(bb0[3],2),round(bb0[4],2))
    o=o2
