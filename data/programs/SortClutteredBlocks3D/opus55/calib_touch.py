from calib_util import *
from calib_grasp import grasp, Rz, M0, ZC
import json, sys
def w2a(b,p):  # world point -> arm frame using guess M0
    return Rz(b[2]).T@(np.array(p)-np.array([b[0],b[1],0]))-M0
def moveto(env,obs,p_world,yawg,steps=250,grip=1.0,tol=0.0015):
    b,q=rstate(obs); a=w2a(b,p_world); a[2]=p_world[2]
    qc,e=ik(q,a,down_R(yawg),0); return joint_ctrl(env,obs,qc,K=1.3,steps=steps,grip=grip,tol=tol)
SAFE=0.33
def touch(env,obs,cube,d,yawg,zlow=ZC+0.006,start=0.04):
    d=np.array([d[0],d[1],0.]); c=objpos(obs,cube)
    p=c-start*d; p[2]=SAFE; obs,_=moveto(env,obs,p,yawg)
    p[2]=zlow; obs,_=moveto(env,obs,p,yawg)
    b,q=rstate(obs); a=w2a(b,p); a[2]=zlow; Rd=down_R(yawg); qc=q.copy()
    da=Rz(b[2]).T@d
    for _ in range(10): obs=env.step(act(dq=np.clip(1.3*(qc-rstate(obs)[1]),-.1,.1),grip=1.0))[0]
    c0=objpos(obs,cube); rec=None
    for i in range(60):
        a=a+0.0015*da; qc,_=ik(qc,a,Rd,0)
        prev=(rstate(obs),objpos(obs,cube))
        obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=3,grip=1.0,tol=0.0)
        c1=objpos(obs,cube)
        if np.linalg.norm((c1-c0)[:2])>0.0004:
            (b,q),cp=prev
            rec=dict(d=d.tolist(),base=b.tolist(),q=q.tolist(),tip=fk_arm(q,0)[0].tolist(),c=c0.tolist(),yawg=yawg); break
    # retreat
    b,q=rstate(obs); a=a-0.03*da; qc,_=ik(qc,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=60,grip=1.0)
    a[2]=SAFE; qc,_=ik(qc,a,Rd,0); obs,_=joint_ctrl(env,obs,qc,K=1.3,steps=100,grip=1.0)
    return obs,rec
def pick_place(env,obs,cube,dst,yawg=0.0):
    c=objpos(obs,cube); p=c.copy(); p[2]=SAFE
    obs,_=moveto(env,obs,p,yawg,grip=0.0)
    for _ in range(8): obs=env.step(act(grip=0.0))[0]
    p[2]=ZC-0.006; obs,_=moveto(env,obs,p,yawg,grip=0.0,tol=0.001)
    for _ in range(12): obs=env.step(act(grip=1.0))[0]
    p[2]=SAFE; obs,_=moveto(env,obs,p,yawg,grip=1.0)
    q=np.array(dst,float); q[2]=SAFE; obs,_=moveto(env,obs,q,yawg,grip=1.0)
    q[2]=ZC+0.012; obs,_=moveto(env,obs,q,yawg,grip=1.0,tol=0.001)
    for _ in range(10): obs=env.step(act(grip=0.0))[0]
    q[2]=SAFE; obs,_=moveto(env,obs,q,yawg,grip=0.0)
    for _ in range(10): obs=env.step(act(grip=1.0))[0]
    return obs
if __name__=='__main__':
    env=make_env(); obs,_=env.reset(seed=0)
    obs,_=drive_base(env,obs,[0.55,-0.012,np.pi])
    obs=pick_place(env,obs,'cube2',[0.2,0.0,0])
    print('placed cube at',objpos(obs,'cube2').round(4))
    poses=json.loads(sys.argv[1]) if len(sys.argv)>1 else [[0.55,0.0,np.pi]]
    recs=[]
    for bp in poses:
        obs,_=drive_base(env,obs,bp,grip=1.0)
        b,_=rstate(obs); yawg=-(wrap(b[2]-np.pi))
        for d in [(1,0),(-1,0),(0,1),(0,-1)]:
            obs,rec=touch(env,obs,'cube2',d,yawg)
            if rec is None: print('no contact',bp,d); continue
            recs.append(rec); b=np.array(rec['base']); t=np.array(rec['tip']); c=np.array(rec['c'])
            pw=b*[1,1,0]+Rz(b[2])@(M0+t)
            print(np.round(bp,3),d,'dot(tip-c,d)',round(float(np.dot(pw-c,rec['d'])),4),'perp',round(float(np.cross(rec['d'],pw-c)[2]),4),'tipz',round(t[2],4))
    json.dump(recs,open(sys.argv[2] if len(sys.argv)>2 else 'calib_touch.json','w'))
