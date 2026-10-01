from probe_lib import *
for lat in [0.09,0.1,0.103,0.107,0.11]:
    o,_=env.reset(seed=11)
    cx,cy,bt=bcenter(o); u=np.array([np.cos(bt),np.sin(bt)]); v=np.array([-u[1],u[0]])
    d=0.18
    o,_=goto(o,*(np.array([cx,cy])-u*(d+0.05)-v*lat),th=bt)
    o,_=goto(o,*(np.array([cx,cy])-u*d-v*lat))
    o,*_=env.step(A(0,0,0,0,1)); b0=obj(o,'target_block'); o,*_=env.step(A(-0.02*u[0],-0.02*u[1],0,0,1))
    print('lat',lat,rob(o)[:2],'moved',np.round(np.array(obj(o,'target_block'))-b0,4))
