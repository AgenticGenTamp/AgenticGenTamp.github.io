import numpy as np, json, robot
_d=json.load(open('env_spaces.json'))
TF={t['name']:t['features'] for t in _d['observation_space']['types']}
RF=TF['mujoco_tidybot_robot']
def J(obs):
    o=obs.get_object_from_name('robot'); return np.array([float(obs.get(o,f)) for f in RF[3:10]])
def B(obs):
    o=obs.get_object_from_name('robot'); return np.array([float(obs.get(o,f)) for f in RF[0:3]])
def P(obs,n): return np.array([float(obs.get(obs.get_object_from_name(n),f)) for f in ['x','y','z']])
class Ctl:
    def __init__(self,env,obs): self.env=env; self.obs=obs; self.grip=0.0; self.steps=0
    def step(self,a):
        self.obs,r,t,tr,inf=self.env.step(a); self.steps+=1; return r,t,tr
    def drive(self,tx,ty,tyaw,n=100,tol=0.004):
        for i in range(n):
            b=B(self.obs); e=np.array([tx-b[0],ty-b[1]]); er=(tyaw-b[2]+np.pi)%(2*np.pi)-np.pi
            a=np.zeros(18,dtype=np.float32); a[0:2]=np.clip(e,-0.1,0.1); a[2]=np.clip(er,-0.1,0.1); a[10]=self.grip
            self.step(a)
            if np.max(np.abs(e))<tol and abs(er)<0.01: break
    def goto(self,qt,n=40,tol=0.01,kv=6.0):
        for i in range(n):
            q=J(self.obs); e=qt-q
            a=np.zeros(18,dtype=np.float32); a[3:10]=np.clip(e,-0.1,0.1); a[11:18]=np.clip(kv*e,-10,10); a[10]=self.grip
            self.step(a)
            if np.max(np.abs(qt-J(self.obs)))<tol: break
    def setgrip(self,g,n=12):
        self.grip=g
        for i in range(n):
            a=np.zeros(18,dtype=np.float32); a[10]=g; self.step(a)

def slow_goto(c,qt,n=120,tol=0.01,rate=0.06,kv=2.0,vmax=0.6):
    import numpy as np
    for i in range(n):
        q=J(c.obs); e=qt-q
        d=np.clip(e,-rate,rate)
        a=np.zeros(18,dtype=np.float32); a[3:10]=d; a[11:18]=np.clip(kv*e,-vmax,vmax); a[10]=c.grip
        c.step(a)
        if np.max(np.abs(qt-J(c.obs)))<tol: break
