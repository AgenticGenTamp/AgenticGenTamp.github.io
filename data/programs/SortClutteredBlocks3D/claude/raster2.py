import numpy as np, sys
from env_client import make_env
import approach as A
from fk import ik, fk

def run(zw, seed=0, half=0.07, dstep=0.008, lines=8, grip=1.0):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    ap=A.GeneratedApproach(None,None,{}); ap.reset(obs,info)
    b,q,g=A.robot_state(obs)
    cb=A.obj_pos(obs,'cube1').copy()
    start=np.array([cb[0], cb[1]-half])
    # phase 1: servo to start at travel height, then descend
    for tw,z,n in [(start, 0.54, 90),(start, zw, 60)]:
        for t in range(n):
            a,gw=ap._servo(b,q,tw,z,grip)
            obs,r,te,tr,i=env.step(a); b,q,g=A.robot_state(obs)
    gw,p=ap.gripper_world(b,q)
    print(' after descend gw',np.round(gw,4),'p',np.round(p,4), 'cube',np.round(A.obj_pos(obs,'cube1'),4))
    qfix=q.copy()
    hits=[]
    def bstep(bt):
        nonlocal b,q,g,obs
        act=np.zeros(11,dtype=np.float32)
        db=np.array(bt)-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        act[0:3]=np.clip(db/0.87,-0.1,0.1)
        dq=(qfix-q+np.pi)%(2*np.pi)-np.pi
        act[3:10]=np.clip(0.45*dq/0.249,-0.1,0.1)
        act[10]=grip
        obs,r,te,tr,i=env.step(act); b,q,g=A.robot_state(obs)
    b0=b.copy()
    for li in range(lines):
        bx = b0[0] - half + li*(2*half/(lines-1))
        ydir = 1 if li%2==0 else -1
        y0 = b0[1] if ydir>0 else b0[1]+2*half
        for t in range(14): bstep(np.array([bx,y0,np.pi]))
        n=int(2*half/dstep)
        for k in range(n+1):
            by = y0 + ydir*k*dstep
            bstep(np.array([bx,by,np.pi]))
            c=A.obj_pos(obs,'cube1')
            if np.linalg.norm(c[:2]-cb[:2])>0.004:
                gw,_=ap.gripper_world(b,q)
                hits.append(('dx_dy_from_cube',round(float(gw[0]-cb[0]),3), round(float(gw[1]-cb[1]),3)))
                env.close(); return hits
    env.close(); return hits
for zw in [float(x) for x in sys.argv[1:]]:
    print('zw',zw, run(zw))
