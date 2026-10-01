import numpy as np, fk, ctrl
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
hist=[]
def step(a):
    global obs
    prev=ctrl.rob(obs)
    obs,r,t,tr,i=env.step(np.asarray(a,dtype=np.float32))
    cur=ctrl.rob(obs)
    return (np.allclose(prev[:10],cur[:10]) and np.abs(np.asarray(a)[:10]).max()>1e-9), t
def move_joints(q_t,grip=0.0,maxit=40,tag=""):
    n=0
    while n<maxit:
        r=ctrl.rob(obs); d=ctrl.wrapd(np.array(q_t)-r[3:10])
        if np.abs(d).max()<1e-5: break
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2); a[10]=grip
        rej,t=step(a); n+=1
        if rej: print("  REJ arm",tag,np.round(ctrl.rob(obs)[3:10],3)); return False,n
    return True,n
def move_base(bt,maxit=40):
    n=0
    while n<maxit:
        r=ctrl.rob(obs); d=np.array(bt)-r[:3]; d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        if np.abs(d).max()<1e-5: break
        a=np.zeros(11); a[:3]=np.clip(d,-0.2,0.2)
        rej,t=step(a); n+=1
        if rej: print("  REJ base",np.round(ctrl.rob(obs)[:3],3)); return False,n
    return True,n
def cart_move(tp,ang,maxit=40,tag="",step_len=0.06):
    """resolved rate toward tool target pos tp (base frame) with orientation targR(ang)"""
    n=0
    R_des=ctrl.targR(ang)
    while n<maxit:
        q=ctrl.rob(obs)[3:10]
        p,R,_=fk.fk_arm(q)
        ep=tp-p; er=fk.so3_error(R,R_des)
        if np.linalg.norm(ep)<2e-3 and np.linalg.norm(er)<1e-2: return True,n
        # sub-target
        sub = p + ep*min(1.0, step_len/max(np.linalg.norm(ep),1e-9))
        qd,ok=fk.ik(sub,R_des,q,iters=60)
        d=ctrl.wrapd(qd-q)
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
        rej,t=step(a); n+=1
        if rej: print("  REJ cart",tag,np.round(p,3)); return False,n
    return False,n
bl=ctrl.blocks(obs)
tb=bl["block1"]; print("target",np.round(tb,3))
p0,_,_=fk.fk_arm(ctrl.rob(obs)[3:10]); print("tool0",np.round(p0,3))
# raise tool
print("raise",cart_move(np.array([p0[0],p0[1],0.95]),np.pi/2,tag="raise"))
bxy=(tb[0]-0.55, tb[1]-0.2)
print("base",move_base((bxy[0],bxy[1],0.0)), np.round(ctrl.rob(obs)[:3],3))
bx,by,_=ctrl.rob(obs)[:3]
ang=ctrl.align_angle(ctrl.quat_yaw(tb[3:7]))
print("ang",ang)
print("above",cart_move(np.array([tb[0]-bx,tb[1]-by,0.95]),ang,tag="above"))
print("down",cart_move(np.array([tb[0]-bx,tb[1]-by,tb[2]+0.02]),ang,tag="down"))
p,_,_=fk.fk_arm(ctrl.rob(obs)[3:10]); print("tool now(base)",np.round(p,4),"world",np.round(p+np.array([bx,by,0]),4))
print("close",step([0]*10+[-1.0]))
print("robot",np.round(ctrl.rob(obs),3))
print("gtf",np.round(ctrl.gtf(obs),4))
env.close()
