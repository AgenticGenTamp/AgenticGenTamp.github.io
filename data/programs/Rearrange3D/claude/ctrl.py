import numpy as np, kinova_fk as K
MOUNT=np.array([0.,0.,0.44])
R_DOWN=np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])
def Rd(yaw=0.0):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@R_DOWN
class Bot:
    def __init__(self, env, seed=0):
        self.env=env
        o,_=env.reset(seed=seed); self.obs=np.asarray(o,float); self.n=0
    def step(self,a):
        o,r,te,tr,i=self.env.step(np.asarray(a,float)); self.obs=np.asarray(o,float); self.n+=1; self.r=r
    def bp(self): return self.obs[93],self.obs[94],self.obs[95]
    def w2r(self,p):
        bx,by,th=self.bp(); c,s=np.cos(-th),np.sin(-th)
        d=np.asarray(p,float)-np.array([bx,by,0.])
        return np.array([c*d[0]-s*d[1],s*d[0]+c*d[1],d[2]])-MOUNT
    def r2w(self,p):
        bx,by,th=self.bp(); p=np.asarray(p,float)+MOUNT; c,s=np.cos(th),np.sin(th)
        return np.array([bx+c*p[0]-s*p[1],by+s*p[0]+c*p[1],p[2]])
    def R_w2r(self,Rw):
        bx,by,th=self.bp(); c,s=np.cos(-th),np.sin(-th)
        return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@Rw
    def ee(self): return self.r2w(K.fk(self.obs[96:103])[:3,3])
    def servo(self, tgt_w, grip, nsteps, Rw=R_DOWN, kp=2.0, kr=1.0, tol=None):
        tr=self.w2r(tgt_w); Rr=self.R_w2r(Rw)
        for t in range(nsteps):
            q=self.obs[96:103].copy(); T=K.fk(q)
            ep=tr-T[:3,3]; er=K.rotation_log(Rr@T[:3,:3].T)
            if tol is not None and np.linalg.norm(ep)<tol: break
            v=np.concatenate([kp*ep, kr*er])
            J=K.jacobian(q); lam=0.08
            dq=J.T@np.linalg.solve(J@J.T+lam*np.eye(6), v)
            dq=np.clip(dq,-0.4,0.4)
            a=np.zeros(11); a[3:10]=np.clip(dq,-0.1,0.1); a[10]=grip
            self.step(a)
        T=K.fk(self.obs[96:103])
        return self.ee(), np.linalg.norm(self.w2r(tgt_w)-T[:3,3])
    def grip(self,g,n=10):
        a=np.zeros(11); a[10]=g
        for _ in range(n): self.step(a)
