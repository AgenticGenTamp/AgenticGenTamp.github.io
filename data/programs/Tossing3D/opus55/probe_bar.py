from env_client import make_env
import numpy as np
env=make_env()
def st(o):
    r=o.get_object_from_name('robot')
    return np.array([o.get(r,'pos_base_x'),o.get(r,'pos_base_y'),o.get(r,'pos_base_rot')])
def goto(env,obs,tx,ty,n=80):
    rs=set()
    for _ in range(n):
        s=st(obs); a=np.zeros(18,np.float32)
        a[0]=np.clip(tx-s[0],-.1,.1); a[1]=np.clip(ty-s[1],-.1,.1); a[2]=np.clip(-s[2],-.1,.1)
        obs,rw,te,tr,inf=env.step(a); rs.add(round(float(rw),3))
    return obs,rs
for y in [0,0.5,1,1.5,2,2.5,3,-1,-2,-3]:
    obs,_=env.reset(seed=0)
    obs,r1=goto(env,obs,0.0,y,60)
    obs,r2=goto(env,obs,3.0,y,60)
    print('y',y,'reached',st(obs).round(3),'rewards',r1|r2)
# arena limits
for tx,ty in [(-10,0),(0,10),(0,-10)]:
    obs,_=env.reset(seed=0)
    obs,_=goto(env,obs,tx,ty,150); print('far',tx,ty,st(obs).round(3))
env.close()
