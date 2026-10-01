import sys
import numpy as np
from env_client import make_env
import kinova_fk as K
np.set_printoptions(precision=4, suppress=True)
MOUNT=np.array([0.,0.,0.44]); SL=32
V=float(sys.argv[1]); OPENV=1.0-V
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
def step(a):
    global obs
    o,r,te,tr,i=env.step(np.asarray(a,float)); obs=np.asarray(o,float)
def w2r(p):
    bx,by,th=obs[93],obs[94],obs[95]; c,s=np.cos(-th),np.sin(-th)
    d=np.asarray(p,float)-np.array([bx,by,0.]); return np.array([c*d[0]-s*d[1],s*d[0]+c*d[1],d[2]])-MOUNT
def r2w(p):
    bx,by,th=obs[93],obs[94],obs[95]; p=np.asarray(p,float)+MOUNT; c,s=np.cos(th),np.sin(th)
    return np.array([bx+c*p[0]-s*p[1],by+s*p[0]+c*p[1],p[2]])
def ee(): return r2w(K.fk(obs[96:103])[:3,3])
def Rd(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])
def servo(tw,n,g,yaw):
    for _ in range(n):
        qt,ok,inf=K.ik(w2r(tw),target_rot=Rd(yaw),q_init=obs[96:103],restarts=1,max_iters=120)
        a=np.zeros(11); a[3:10]=np.clip((qt-obs[96:103])*4.0,-0.1,0.1); a[10]=g
        step(a)
    return ee()
can0=obs[SL:SL+3].copy(); print("can",can0)
for yaw in [0.0, np.pi/2]:
    for dz in [0.0,-0.04]:
        servo(can0+np.array([-0.15,0,0.22]),60,OPENV,yaw)
        e=servo(can0+np.array([-0.15,0,dz]),70,OPENV,yaw)
        pb=obs[SL:SL+3].copy()
        # lateral entry
        for k in range(1,16):
            e=servo(can0+np.array([-0.15+0.01*k,0,dz]),6,OPENV,yaw)
        d_enter=np.linalg.norm(obs[SL:SL+3]-pb)
        p2=obs[SL:SL+3].copy(); z2=obs[SL+2]
        a=np.zeros(11); a[10]=V
        for _ in range(12): step(a)
        d_close=np.linalg.norm(obs[SL:SL+3]-p2)
        e2=servo(can0+np.array([0,0,0.25]),70,V,yaw)
        dzl=obs[SL+2]-z2
        print("V=%.1f yaw=%.2f dz=%+.2f | EEentry=%s err=%.3f | enter_move=%.4f close_move=%.4f LIFT_dz=%+.4f can=%s"%(
            V,yaw,dz,e,np.linalg.norm(e-(can0+np.array([0,0,dz]))),d_enter,d_close,dzl,obs[SL:SL+3]))
        if dzl>0.03:
            print("*** GRASP SUCCESS V=%.1f yaw=%.2f dz=%+.2f"%(V,yaw,dz)); env.close(); sys.exit()
        a=np.zeros(11); a[10]=OPENV
        for _ in range(6): step(a)
env.close()
