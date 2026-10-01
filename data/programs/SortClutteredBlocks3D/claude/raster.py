import numpy as np, sys
from env_client import make_env
import approach as A
from fk import ik, fk

def run(z_arm, seed=0, half=0.07, dstep=0.01, lines=8):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':4})
    b,q,g=A.robot_state(obs)
    cb=A.obj_pos(obs,'cube1').copy()
    R_NOM=0.62
    qt=ik(np.array([R_NOM,0.0,z_arm]), A.tool_R(0.0), q)
    # move arm to pose and base to start of raster, gripper closed
    def step(bt,grip=1.0):
        nonlocal b,q,g,obs
        a=A.GeneratedApproach.__new__(A.GeneratedApproach)
        act=np.zeros(11,dtype=np.float32)
        db=np.array(bt)-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
        act[0:3]=np.clip(db/0.87,-0.1,0.1)
        dq=(qt-q+np.pi)%(2*np.pi)-np.pi
        act[3:10]=np.clip(0.45*dq/0.249,-0.1,0.1)
        act[10]=grip
        obs,r,te,tr,i=env.step(act); b,q,g=A.robot_state(obs)
    start=np.array([cb[0]+R_NOM, cb[1]-half, np.pi])
    for t in range(160): step(start)
    p=fk(q)[:3,3]
    print('arm p',np.round(p,4),'base',np.round(b,3))
    hits=[]
    for li in range(lines):
        bx = cb[0]+R_NOM - half + li*(2*half/(lines-1))
        ydir = 1 if li%2==0 else -1
        y0 = cb[1]-half*ydir
        for t in range(160): pass
        # move to line start
        for t in range(12): step(np.array([bx,y0,np.pi]))
        n=int(2*half/dstep)
        for k in range(n+1):
            by = y0 + ydir*k*dstep
            step(np.array([bx,by,np.pi]))
            c=A.obj_pos(obs,'cube1')
            if np.linalg.norm(c[:2]-cb[:2])>0.004:
                gw=np.array([b[0]-fk(q)[0,3], b[1]-fk(q)[1,3]])
                hits.append((round(float(gw[0]-cb[0]),3), round(float(gw[1]-cb[1]),3), round(float(b[0]),3), round(float(b[1]),3)))
                env.close(); return p, hits
    env.close(); return p, hits
for z in [float(x) for x in sys.argv[1:]]:
    print('z_arm',z, run(z))
