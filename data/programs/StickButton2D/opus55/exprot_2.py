from exprot import *
def corners(s,w=0.05,h=1.25):
    x,y,th=s; c,sn=math.cos(th),math.sin(th)
    return [(x+c*u-sn*v, y+sn*u+c*v) for u,v in ((0,0),(w,0),(w,h),(0,h))]
def bb(s):
    P=corners(s); xs=[p[0] for p in P]; ys=[p[1] for p in P]
    return 'X[%.4f,%.4f] Y[%.4f,%.4f]'%(min(xs),max(xs),min(ys),max(ys))
def bbc(s):  # center-anchored alternative
    x,y,th=s; c,sn=math.cos(th),math.sin(th); P=[(x+c*u-sn*v,y+sn*u+c*v) for u,v in ((-.025,-.625),(.025,-.625),(.025,.625),(-.025,.625))]
    xs=[p[0] for p in P]; ys=[p[1] for p in P]; return 'centerconv X[%.4f,%.4f] Y[%.4f,%.4f]'%(min(xs),max(xs),min(ys),max(ys))
def acc(o,o2):
    r,s,_=rd(o); r2,s2,_=rd(o2)
    return max(abs(r2['x']-r['x']),abs(r2['y']-r['y']),abs(wrap(r2['theta']-r['theta'])))>1e-7
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
obs,acts=to_grasp(seed); r,s,b=rd(obs); print('grasp robot',r,'stick',s, bb(s))
# fine rotation to wall, both directions
for sgn in (1,-1):
    o=replay(seed,acts); tot=0
    for i in range(700):
        o2=step(o,dth=sgn*0.005)
        if not acc(o,o2):
            r,s,_=rd(o); p=pred(r,s,sgn*0.005)
            print('sgn',sgn,'rejected after',round(tot,3),'last ok',bb(rd(o)[1]),'| rejected pose',bb(p),'|',bbc(p)); break
        tot+=sgn*0.005; o=o2
    else: print('sgn',sgn,'never rejected, total',tot)
