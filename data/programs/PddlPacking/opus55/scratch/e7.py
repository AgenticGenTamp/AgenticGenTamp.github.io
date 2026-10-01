import numpy as np
exec(open('scratch/e5.py').read().split('P=np.pi')[0])
P=np.pi
for x in [-0.45,-0.5,-0.55,-0.6]:
    a=bis([(-1,0,0),(x,0,0)],lambda v:(x,v,0),0,0.9); b=bis([(-1,0,0),(x,0,0)],lambda v:(x,v,0),0,-0.9)
    print("x",x,"yaw0 y range",round(b,4),round(a,4))
# swept check: jump through table in one step
o,_=env.reset(seed=0); s=rs(o); a=np.zeros(11,np.float32); a[0]=2.0
o2,*_=env.step(a); print("jump -1->+1:",rs(o2)[:3])
o,_=env.reset(seed=0); a=np.zeros(11,np.float32); a[0]=0.75
o2,*_=env.step(a); print("jump -1->-0.25:",rs(o2)[:3])
# rotate at limit
o,_=env.reset(seed=0); o,ok=goto(o,(-0.43,0,0)); print(ok)
for yaw in [0.1,0.3,0.5,-0.1,-0.3]:
    o2,ok=goto(o,(-0.43,0,yaw),maxd=0.01); print("rot at -0.43 to",yaw,ok)
