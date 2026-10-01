import numpy as np, fk, ctrl
class Runner:
    def __init__(self, env, seed=0):
        self.env=env
        self.obs,self.info=env.reset(seed=seed)
        self.n=0
    def step(self,a):
        prev=ctrl.rob(self.obs)
        self.obs,r,t,tr,i=self.env.step(np.asarray(a,dtype=np.float32))
        self.n+=1
        cur=ctrl.rob(self.obs)
        rej = np.allclose(prev[:10],cur[:10]) and np.abs(np.asarray(a)[:10]).max()>1e-9
        return rej,t
    def rob(self): return ctrl.rob(self.obs)
    def tool(self):
        r=self.rob(); return fk.fk_arm(r[3:10])[0]
    def move_base(self,bt,maxit=40):
        for n in range(maxit):
            r=self.rob(); d=np.array(bt)-r[:3]; d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
            if np.abs(d).max()<1e-5: return True
            a=np.zeros(11); a[:3]=np.clip(d,-0.2,0.2)
            rej,t=self.step(a)
            if rej: return False
        return False
    def move_joints(self,q_t,grip=0.0,maxit=40):
        for n in range(maxit):
            r=self.rob(); d=ctrl.wrapd(np.array(q_t)-r[3:10])
            if np.abs(d).max()<1e-5: return True
            a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2); a[10]=grip
            rej,t=self.step(a)
            if rej: return False
        return True
    def cart(self,tp,ang,maxit=60,step_len=0.05,tol=2e-3):
        R_des=ctrl.targR(ang)
        for n in range(maxit):
            q=self.rob()[3:10]; p,R,_=fk.fk_arm(q)
            ep=np.asarray(tp)-p; er=fk.so3_error(R,R_des)
            if np.linalg.norm(ep)<tol and np.linalg.norm(er)<1e-2: return True
            sub=p+ep*min(1.0,step_len/max(np.linalg.norm(ep),1e-9))
            qd,ok=fk.ik(sub,R_des,q,iters=60)
            d=ctrl.wrapd(qd-q)
            if np.abs(d).max()<1e-6: return False
            a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
            rej,t=self.step(a)
            if rej: return False
        return False
