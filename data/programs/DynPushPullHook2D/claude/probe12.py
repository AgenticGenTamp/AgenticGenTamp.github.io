from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
def gotoseq(env,obs,step,seq): pass
for seed in [42,0,7]:
    obs,info=env.reset(seed=seed)
    def step(a):
        global obs
        obs,_,_,_,_=env.step(a)
    def H(): return (round(oget(obs,'hook','x'),3),round(oget(obs,'hook','y'),3),round(oget(obs,'hook','theta'),3),oget(obs,'hook','held'))
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)])
    base=np.array([hx,hy])-1.55*u
    # open gripper + rotate + retract
    for _ in range(60):
        d=(hth-rget(obs,'theta')+np.pi)%(2*np.pi)-np.pi
        step(act(dth=d,dg=0.02,da=-0.1))
    # travel at safe y (below hook) to base x
    safey=0.26
    for _ in range(400):
        dx=np.clip(base[0]-rget(obs,'x'),-0.05,0.05); dy=np.clip(safey-rget(obs,'y'),-0.05,0.05)
        if abs(dx)<0.004 and abs(dy)<0.004: break
        step(act(dx=dx,dy=dy))
    for _ in range(400):
        dy=np.clip(base[1]-rget(obs,'y'),-0.05,0.05)
        if abs(dy)<0.004: break
        step(act(dy=dy))
    print(seed,"pre-close robot",round(rget(obs,'x'),3),round(rget(obs,'y'),3),"hook",H())
    for _ in range(30): step(act(dg=-0.02))
    print(seed,"after close hook",H())
    for _ in range(20): step(act(dx=-0.05,dy=0.05))
    print(seed,"after move robot",round(rget(obs,'x'),3),round(rget(obs,'y'),3),"hook",H())
env.close()
