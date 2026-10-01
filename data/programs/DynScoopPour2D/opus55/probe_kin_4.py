import numpy as np
from env_client import make_env
env=make_env()
def R(o): r=o.get_object_from_name('robot'); return np.array([o.get(r,f) for f in ['x','y','theta','arm_joint','finger_gap']])
def st(a):
    global obs; obs=env.step(np.array(a,dtype=float))[0]
def go(tx,ty,tth=None,n=300):
    for i in range(n):
        p=R(obs); dx=np.clip(tx-p[0],-.03,.03); dy=np.clip(ty-p[1],-.03,.03)
        dth=0 if tth is None else np.clip(tth-p[2],-.098,.098)
        if abs(dx)<1e-6 and abs(dy)<1e-6 and abs(dth)<1e-6: return True
        st([dx,dy,dth,0,0])
        if np.allclose(R(obs),p): return False
    return False
def push(a,n=300):
    for i in range(n):
        p=R(obs); st(a)
        if np.allclose(R(obs),p): break
    return np.round(R(obs),4)
obs,_=env.reset(seed=0)
print(go(2.09,2.5,np.pi/2), np.round(R(obs),3))  # arm up
print(go(1.0,2.5), go(1.0,1.0), np.round(R(obs),3))
print('push right at y=1.0 arm up:',push([0.03,0,0,0,0]))
print('push right fine:',push([0.001,0,0,0,0]))
go(R(obs)[0]-0.1,1.0); print(go(1.0,2.5),go(2.8,2.5),go(2.8,1.0),np.round(R(obs),3))
print('push left at y=1.0 arm up:',push([-0.03,0,0,0,0])); print('fine',push([-0.001,0,0,0,0]))
go(R(obs)[0]+0.1,1.0); go(2.8,2.5); go(1.9,2.5)
print('push down over wall arm up:',push([0,-0.03,0,0,0])); print('fine',push([0,-0.001,0,0,0]))
# bottom / left bounds, arm up
go(2.8,2.5); print('down right side',push([0,-0.03,0,0,0]),push([0,-0.001,0,0,0]))
print('right',push([0.03,0,0,0,0]),push([0.001,0,0,0,0]))
go(R(obs)[0],2.5); print('up',push([0,0.03,0,0,0]),push([0,0.001,0,0,0]))
go(1.0,2.5); print('left',push([-0.03,0,0,0,0]),push([-0.001,0,0,0,0]))
