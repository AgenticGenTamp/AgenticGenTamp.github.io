import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
log=[]
def step(a):
    global obs
    prev=ctrl.rob(obs)
    obs,r,t,tr,i=env.step(np.asarray(a,dtype=np.float32))
    cur=ctrl.rob(obs)
    rej = np.allclose(prev[:10],cur[:10]) and np.abs(a[:10]).max()>1e-9
    return rej,t
def move_arm(q_t,grip=0.0,maxit=60):
    n=0
    while n<maxit:
        r=ctrl.rob(obs); d=ctrl.wrapd(np.array(q_t)-r[3:10])
        if np.abs(d).max()<1e-5: break
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2); a[10]=grip
        rej,t=step(a); n+=1
        if rej: print("  arm rejected at",np.round(ctrl.rob(obs)[3:10],3)); return False,n
    return True,n
def move_base(bt,maxit=60):
    n=0
    while n<maxit:
        r=ctrl.rob(obs); d=np.array(bt)-r[:3]
        d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        if np.abs(d).max()<1e-5: break
        a=np.zeros(11); a[:3]=np.clip(d,-0.2,0.2)
        rej,t=step(a); n+=1
        if rej: print("  base rejected at",np.round(ctrl.rob(obs)[:3],3)); return False,n
    return True,n
# carry pose: tool high
q0=ctrl.rob(obs)[3:10]
qc,ok=fk.ik(np.array([0.25,0.35,1.15]),ctrl.targR(np.pi/2),q0)
print("carry ik",ok,np.round(qc,3), np.round(fk.fk_arm(qc)[0],3))
print(move_arm(qc))
print("base fwd:",move_base((-0.45,0,0)))
print("base fwd more:",move_base((-0.30,0,0)))
print("pos",np.round(ctrl.rob(obs)[:3],3))
print("base y:",move_base((ctrl.rob(obs)[0],-0.5,0)))
print("pos",np.round(ctrl.rob(obs)[:3],3))
env.close()
