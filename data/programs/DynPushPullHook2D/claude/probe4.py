from env_client import make_env
import numpy as np
env = make_env()
obs,info = env.reset(seed=42)
def R(o,f): 
    return float(o.get(o.get_object_from_name("robot"),f))
def go(dx,dy,dth,da,dg,n):
    global obs
    for _ in range(n):
        obs,_,_,_,_ = env.step(np.array([dx,dy,dth,da,dg],dtype=np.float32))
# arm limits
go(0,0,0,-0.0999,0,30); print("arm min", R(obs,'arm_length'))
go(0,0,0,0.0999,0,60); print("arm max", R(obs,'arm_length'))
go(0,0,0,0,0.0199,60); print("grip max", R(obs,'finger_gap'))
go(0,0,0,0,-0.0199,60); print("grip min", R(obs,'finger_gap'))
go(0,0,0,-0.0999,0,30)
# move to left bottom
go(-0.0499,-0.0499,0,0,0,200); print("min xy", R(obs,'x'), R(obs,'y'))
# sweep x, testing y max
for tx in np.arange(0.3,3.4,0.3):
    # move down first
    go(0,-0.0499,0,0,0,60)
    cur=R(obs,'x')
    d=tx-cur
    n=int(abs(d)/0.0499)+1
    go(np.sign(d)*0.0499,0,0,0,0,n)
    go(0,0.0499,0,0,0,60)
    print(round(R(obs,'x'),3), round(R(obs,'y'),3))
env.close()
