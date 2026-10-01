import numpy as np, fk, ctrl, lib
from env_client import make_env
def targR_tilt(ang, tilt, tiltdir):
    x=np.array([np.sin(tilt)*np.cos(tiltdir), np.sin(tilt)*np.sin(tiltdir), -np.cos(tilt)])
    y=np.array([np.cos(ang),np.sin(ang),0.0]); y=y-x*np.dot(x,y); y/=np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])
env=make_env(); R=lib.Runner(env,seed=5)
bl=ctrl.blocks(R.obs); tb=bl["block2"]; print("block2",np.round(tb,3))
print("base",R.move_base((-0.43,-0.22,0.0)),np.round(R.rob()[:3],3))
byaw=2*np.arctan2(tb[5],tb[6])
for tilt in [0.35,0.5]:
  for tiltdir in [-np.pi/2, np.pi]:
    ang=ctrl.align_angle(byaw,np.pi/2)
    Rd=targR_tilt(ang,tilt,tiltdir)
    # approach: pre pose 0.12 back along approach from grasp point
    ap=Rd[:,0]
    gp=np.array([tb[0],tb[1],tb[2]])-0.05*ap
    pre=gp-0.12*ap
    bx,by,_=R.rob()[:3]
    def cart_to(pw,Rd,maxit=40,step=0.05):
        for i in range(maxit):
            q=R.rob()[3:10]; p,Rc,_=fk.fk_arm(q)
            tp=pw-np.array([bx,by,0]); ep=tp-p; er=fk.so3_error(Rc,Rd)
            if np.linalg.norm(ep)<3e-3 and np.linalg.norm(er)<1e-2: return True
            sub=p+ep*min(1.0,step/max(np.linalg.norm(ep),1e-9))
            qd,ok=fk.ik(sub,Rd,q,iters=60)
            d=ctrl.wrapd(qd-q)
            if np.abs(d).max()<1e-7: return False
            a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
            rej,t=R.step(a)
            if rej: return False
        return False
    ok1=cart_to(np.array([pre[0],pre[1],max(pre[2],0.95)]),Rd)
    ok2=cart_to(pre,Rd); ok3=cart_to(gp,Rd,step=0.03)
    rej,t=R.step([0]*10+[-1.0]); ga=R.rob()[11]
    print("tilt",tilt,"dir",round(tiltdir,2),ok1,ok2,ok3,"grasp",ga,np.round(ctrl.gtf(R.obs),4))
    if ga:
        # lift and check block pose
        p,Rc,_=fk.fk_arm(R.rob()[3:10])
        cart_to(np.array([tb[0],tb[1],1.0]),Rd)
        print("block after lift",np.round(ctrl.blocks(R.obs)["block2"][:7],3))
        break
    R.step([0]*10+[1.0])
env.close()
