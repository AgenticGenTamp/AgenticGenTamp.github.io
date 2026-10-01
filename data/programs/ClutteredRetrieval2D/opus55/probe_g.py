from probe_lib import *
# grasp distance test, seed 11
for gap in [0.0005,0.012,0.014,0.0149,0.0151,0.016,0.018]:
    o,_=env.reset(seed=11)
    cx,cy,bt=bcenter(o)
    # face +x in block frame: approach from block's local -x side
    u=np.array([np.cos(bt),np.sin(bt)])
    # gripper front face at aj+0.005 from robot center; block face at 0.07 from center
    d=0.07+0.005+0.1+gap
    p=np.array([cx,cy])-u*(d+0.05)
    o,ok=goto(o,*p,th=bt); 
    p=np.array([cx,cy])-u*d
    o,ok=goto(o,*p); r0=rob(o)
    o,*_=env.step(A(0,0,0,0,1)); b0=obj(o,'target_block')
    o,*_=env.step(A(-0.03*u[0],-0.03*u[1],0,0,1)); b1=obj(o,'target_block')
    print('gap',gap,'ok',ok,'rob',r0,'block moved',np.round(np.array(b1)-np.array(b0),4), 'robot moved', np.round(np.array(rob(o)[:2])-np.array(r0[:2]),4))
