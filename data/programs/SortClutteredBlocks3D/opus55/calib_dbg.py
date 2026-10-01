from calib_touch import *
import calib_touch as T
env=make_env(); obs,_=env.reset(seed=0)
obs,_=drive_base(env,obs,[0.55,-0.012,np.pi]); obs=pick_place(env,obs,'cube2',[0.2,0.0,0])
obs,_=drive_base(env,obs,[0.7,0,np.pi],grip=1.0)
d=np.array([-1.,0,0]); c=objpos(obs,'cube2')+[0,0.06,0]; yawg=0.0
p=c-0.04*d; p[2]=SAFE; obs,_=moveto(env,obs,p,yawg)
p[2]=ZC+0.006; obs,_=moveto(env,obs,p,yawg)
b,q=rstate(obs); a=w2a(b,p); a[2]=ZC+0.006; Rd=down_R(yawg); qc=q.copy(); da=Rz(b[2]).T@d
print('cube',objpos(obs,'cube2').round(4))
for i in range(30):
    a=a+0.0015*da; qc,_=ik(qc,a,Rd,0)
    for k in range(3):
        bq=rstate(obs); obs=env.step(act(dq=np.clip(1.3*(qc-bq[1]),-.1,.1),grip=1.0))[0]
    b,q=rstate(obs); t,R,_,_=fk_arm(q,0); pw=b*[1,1,0]+Rz(b[2])@(M0+t)
    print('qerr',(qc-q).round(3),'q',q.round(3)) if i in (9,12,16) else None
    print(i,'tipw',pw.round(4),'Rz',R[:,2].round(3),'cube',objpos(obs,'cube2').round(4),'base',b.round(4))
