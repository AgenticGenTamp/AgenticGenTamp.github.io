import numpy as np, kin
JN=["joint_%d"%i for i in range(1,8)]
POSF=["pose_x","pose_y","pose_z"]
MZ=0.245; FWD=0.12
class Ctl:
    def __init__(self, env, obs):
        self.env=env; self.o=obs; self.steps=0; self.done=False
    def robot(self):
        o=self.o; R=o.get_object_from_name("robot")
        b=np.array([float(o.get(R,f)) for f in ["pos_base_x","pos_base_y","pos_base_rot"]])
        q=np.array([float(o.get(R,j)) for j in JN])
        return b,q,float(o.get(R,"grasp_active"))
    def opos(self,n):
        o=self.o; ob=o.get_object_from_name(n)
        return np.array([float(o.get(ob,f)) for f in POSF])
    def step(self,a):
        self.o,r,t,tr,info=self.env.step(np.asarray(a,dtype=np.float32)); self.steps+=1
        if t: self.done=True
        return t,tr
    def armbase(self,b=None):
        if b is None: b,_,_=self.robot()
        return b[0]+FWD*np.cos(b[2]), b[1]+FWD*np.sin(b[2])
    def gotobase(self,tx,ty,trot,nmax=80):
        for k in range(nmax):
            b,q,g=self.robot()
            dr=(trot-b[2]+np.pi)%(2*np.pi)-np.pi
            e=np.array([tx-b[0],ty-b[1],dr])
            if np.abs(e).max()<1e-4: return True
            a=np.zeros(11); a[:3]=np.clip(e,-0.2,0.2)
            prev=(b.copy())
            t,tr=self.step(a)
            b2,_,_=self.robot()
            if np.allclose(b2,prev): return False
        return False
    def goto(self,qd,nmax=60,tol=1e-4):
        for k in range(nmax):
            b,q,g=self.robot(); e=qd-q
            if np.abs(e).max()<tol: return True
            a=np.zeros(11); a[3:10]=np.clip(e,-0.2,0.2)
            qp=q.copy(); self.step(a); b,q,g=self.robot()
            if np.allclose(q,qp): return False
        return False
    def unwrap(self,qd,q=None):
        if qd is None: return None
        if q is None: b,q,g=self.robot()
        qd=np.array(qd,dtype=float)
        for i in range(7):
            best=qd[i]
            for k in (-2,-1,0,1,2):
                cand=qd[i]+2*np.pi*k
                if kin.JOINT_LIMITS[i,0]-1e-9<=cand<=kin.JOINT_LIMITS[i,1]+1e-9 and abs(cand-q[i])<abs(best-q[i]):
                    best=cand
            qd[i]=best
        return qd
    def ik(self,pos,yaw=None,mz=MZ,**kw):
        b,q,g=self.robot(); ax,ay=self.armbase(b)
        if yaw is None: yaw=b[2]
        return self.unwrap(kin.ik_top_down(np.asarray(pos),yaw=yaw,q_init=q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,mz),**kw),q)
    def ikR(self,pos,R,mz=MZ,**kw):
        b,q,g=self.robot(); ax,ay=self.armbase(b)
        return self.unwrap(kin.ik(np.asarray(pos),target_R=R,q_init=q,base_x=ax,base_y=ay,base_rot=b[2],mount=(0,0,mz),**kw),q)
    def move_R(self,pos,R,mz=MZ,**kw):
        qd=self.ikR(pos,R,mz,**kw)
        if qd is None: return False
        return self.goto(qd)
    def move_to(self,pos,yaw=None,mz=MZ,tol=None,gtol=1e-4,**kw):
        if tol is not None: kw['pos_tol']=tol; kw.setdefault('rot_tol',1e-3); kw.setdefault('max_iters',200)
        qd=self.ik(pos,yaw,mz,**kw)
        if qd is None: return False
        return self.goto(qd,tol=gtol)
    def grip(self,close=True):
        a=np.zeros(11); a[10]=-1.0 if close else 1.0
        self.step(a)
