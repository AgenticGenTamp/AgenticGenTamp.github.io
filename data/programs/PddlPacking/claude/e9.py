import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
def step(a):
    global obs
    prev=ctrl.rob(obs)
    obs,r,t,tr,i=env.step(np.asarray(a,dtype=np.float32))
    cur=ctrl.rob(obs)
    return (np.allclose(prev[:10],cur[:10]) and np.abs(np.asarray(a)[:10]).max()>1e-9), t
def move(qb=None,q_t=None,grip=0.0,maxit=60,verbose=True):
    n=0
    while n<maxit:
        r=ctrl.rob(obs); a=np.zeros(11); d=np.zeros(10)
        if qb is not None: d[:3]=np.array(qb)-r[:3]
        if q_t is not None: d[3:]=ctrl.wrapd(np.array(q_t)-r[3:10])
        if np.abs(d).max()<1e-5:
            if grip!=0: step([0]*10+[grip])
            break
        a[:10]=np.clip(d,-0.2,0.2); a[10]=grip
        rej,t=step(a); n+=1
        if rej:
            if verbose: print("  rejected",np.round(ctrl.rob(obs)[:10],3))
            return False,n
    return True,n
bl=ctrl.blocks(obs)
tb=bl["block0"]; print("block0",np.round(tb,3))
print(move(q_t=[0.994,-0.524,0.893,-0.723,0.771,-2.094,2.373]))
print(move(qb=(-0.45,0,0)))
bx,by,_=ctrl.rob(obs)[:3]
ang=ctrl.align_angle(ctrl.quat_yaw(tb[3:7]))
for dz in [0.10, 0.02]:
    tp=np.array([tb[0]-bx,tb[1]-by,tb[2]+dz])
    q,ok=fk.ik(tp,ctrl.targR(ang),ctrl.rob(obs)[3:10])
    print("ik",dz,ok,np.round(fk.fk_arm(q)[0]-tp,4))
    print(move(q_t=q))
print("robot",np.round(ctrl.rob(obs),3))
print("close:",step([0]*10+[-1.0]))
print("robot",np.round(ctrl.rob(obs),3))
print("gtf",np.round(ctrl.gtf(obs),4))
print("blocks",{k:np.round(v,3) for k,v in ctrl.blocks(obs).items()})
env.close()
