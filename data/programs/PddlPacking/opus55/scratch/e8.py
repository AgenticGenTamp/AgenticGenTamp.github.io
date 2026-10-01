import numpy as np
exec(open('scratch/e5.py').read().split('P=np.pi')[0])
P=np.pi
for yaw in [0.05,0.1,0.2,0.3,-0.05,-0.1,-0.2]:
    print("at (-0.43,0) rotate to",yaw, feas([(-1,0,0),(-0.43,0,0)],(-0.43,0,yaw)), " at -0.5:",feas([(-1,0,0),(-0.5,0,0)],(-0.5,0,yaw)))
for x in [-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6]:
    v=bis([(-1,-1.5,P/2),(x,-1.5,P/2)],lambda v:(x,v,P/2),-1.5,0.0)
    print("from -y facing table, x=",x," y limit",round(v,4))
