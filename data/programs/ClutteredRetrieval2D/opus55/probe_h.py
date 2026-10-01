from probe_lib import *
def grasp(seed,vacv=1.0,gap=0.005):
    o,_=env.reset(seed=seed)
    cx,cy,bt=bcenter(o); u=np.array([np.cos(bt),np.sin(bt)])
    d=0.07+0.005+0.1+gap
    o,_=goto(o,*(np.array([cx,cy])-u*(d+0.05)),th=bt)
    o,_=goto(o,*(np.array([cx,cy])-u*d))
    o,*_=env.step(A(0,0,0,0,vacv)); return o
def rel(o):
    r=rob(o); b=bcenter(o); th=r[2]
    dx,dy=b[0]-r[0],b[1]-r[1]
    return np.round([np.cos(th)*dx+np.sin(th)*dy,-np.sin(th)*dx+np.cos(th)*dy,wrap(b[2]-th)],4)
for v in [0.4,0.5,0.51]:
    o=grasp(11,v); b0=obj(o,'target_block'); o,*_=env.step(A(-0.03,0,0,0,v))
    print('vac',v,'block moved', np.round(np.array(obj(o,'target_block'))-b0,4))
o=grasp(11); print('rel0',rel(o))
o,*_=env.step(A(-0.05,0,0,0,1)); print('after move rel',rel(o))
o,*_=env.step(A(0,0,0.196,0,1)); print('after rot',rel(o), rob(o))
o,*_=env.step(A(0,0,0,0.05,1)); print('after arm+0.05',rel(o), rob(o))
o,*_=env.step(A(0.0,0.0,0,0,0)); b0=obj(o,'target_block'); print('released rel',rel(o))
o,*_=env.step(A(-0.05,0,0,0,0)); print('after release move, block delta',np.round(np.array(obj(o,'target_block'))-b0,4))
# held-object collision: grasp, then move toward obstruction0 (+y)
o=grasp(11); print('regrasp rob',rob(o),'obs0',obj(o,'obstruction0'))
for i in range(20):
    r=rob(o); o,*_=env.step(A(0,0.01,0,0,1))
    if rob(o)==r: print('blocked at step',i,rob(o),'block',bcenter(o)); break
