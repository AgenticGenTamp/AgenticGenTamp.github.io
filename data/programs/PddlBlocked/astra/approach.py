import math
import numpy as np
from kinematics import fk, solve

class GeneratedApproach:
    """Calibrated side grasps with short base routes and observed collision recovery."""
    def __init__(self,action_space,observation_space,primitives):
        self.space=observation_space
        self.fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
    def pos(self,s,name):
        o=s.get_object_from_name(name);return np.array([s.get(o,'pose_'+f) for f in 'xyz'])
    def reset(self,s,info):
        self.goalname='green0';self.r=s.get_object_from_name('robot');self.cleared=False;self.stage=0;self.queue=[];self.ticks=0;self.prev=None;self.lastmoving=False;self.stuck=0
        self.g=self.pos(s,'green0');self.b=self.pos(s,'blocker');self.d=(self.g-self.b)/np.linalg.norm(self.g-self.b);self.yaw=math.atan2(self.d[1],self.d[0])
        # First preserve the arm pose and translate along the pen opening.
        self.candidates=[]
        for xy in [[x,y] for x,y in [(3.65,0),(3.73,-.4),(3.55,0),(3.73,self.g[1]),(3.73,self.g[1]-.3),(3.73,self.g[1]+.3),(3.65,self.g[1]+.3)]]+[[self.g[0]+v,y] for y in [1.10,1.14,1.06] for v in [0,-.35,.35]]:
            if xy[0]>4.98:continue
            cfg,err=solve(self.g-.035*self.d+np.array([0,0,.03]),self.yaw,xy)
            cr=cfg.copy();cr[:2]-=.15*self.d[:2]
            cp=cfg.copy();cp[:2]-=.32*self.d[:2]
            def safe(v):return abs(v[0])<4.99 and abs(v[1])<4.99 and (v[0]<3.79 or v[1]>.99 or v[1]<-.99)
            if err<.005 and all(safe(v) for v in [cfg,cr,cp]):
                self.candidates.append((cfg,cr,cp))
        # Fixed-base arm motion covers openings where translating the base collides.
        for xy in [[self.g[0]+dx,y] for y in [1.02,1.08,1.15] for dx in [.35,.15,0,-.2]]:
            if xy[0]>4.99:continue
            cr,er=solve(self.b-.035*self.d+np.array([0,0,.03]),self.yaw,xy)
            cg,eg=solve(self.g-.035*self.d+np.array([0,0,.03]),self.yaw,xy,initial=cr)
            if max(er,eg)<.005:self.candidates.append((cg,cr,cr.copy()))
        # A shoulder facing the table reaches around difficult southeast openings.
        if len(self.candidates)<8:
            seed=np.array([-1.57079633,1.05633497,.86637794,2.79058561,-1.54958562,-1.25559226,-.25289311,-3.10850977])
            for xy in [[3.78,.2],[3.74,.2],[3.78,0],[3.74,0],[3.78,.4],[3.7,.4],[3.795,.1]]:
                cr,er=solve(self.b-.035*self.d+np.array([0,0,.03]),self.yaw,xy,initial=seed)
                cg,eg=solve(self.g-.035*self.d+np.array([0,0,.03]),self.yaw,xy,initial=cr)
                if max(er,eg)<.005:self.candidates.append((cg,cr,cr.copy()))
        if not self.candidates:
            seed=np.array([-1.57079633,.7910053262,.9834598963,.7341987732,-.4229021284,2.3241283275,-1.2178303869,-.3902997021])
            for xy in [[3.795,0],[3.795,.1],[3.78,.2]]:
                cr,er=solve(self.b-.02*self.d+np.array([0,0,.06]),self.yaw,xy,initial=seed)
                cg,eg=solve(self.g-.035*self.d+np.array([0,0,.03]),self.yaw,xy,initial=cr)
                if max(er,eg)<.005:self.candidates.append((cg,cr,cr.copy()))
        self.spares=[o.name for o in s.get_objects(self.space.get_type('block')) if o.name.startswith('green') and o.name!='green0']
        self.attempt=0;self.setup(s)
    def add(self,v,grip=0):self.queue.append((np.array(v).copy(),grip))
    def setup(self,s):
        p=np.array([s.get(self.r,f) for f in self.fs]);self.queue=[];self.stage=0
        if self.spares and (not self.candidates or self.candidates[0][2][1]>.95 or self.attempt>2):
            self.setup_spare(s,p);return
        if not self.candidates:
            self.add(p);return
        self.cg,self.cr,self.cp=self.candidates[self.attempt%len(self.candidates)]
        self.base=self.cp[:2].copy();self.north=self.base[1]>.95
        self.high=self.cp.copy();self.high[4]-=.35 if np.max(abs(self.cp-self.cr))<1e-6 else .3
        v=p.copy()
        if p[0]>3.8 and p[1]>.75:v[1]=2.;self.add(v,1)
        v[0]=3.1;self.add(v,1)
        if self.north:v[1]=2. if self.cleared else 1.4;self.add(v,1)
        else:v[1]=self.base[1];self.add(v,1)
        v[2:]=self.high[2:];self.add(v,1)
        if self.north:v[0]=self.base[0];self.add(v,1)
        v[:2]=self.base;self.add(v,1)
        self.add(self.high,1);self.add(self.cp,1);self.add(self.cr,1 if self.cleared else -1)
        if self.cleared:self.stage=2;self.add(self.cg,-1)
    def setup_spare(self,s,p):
        self.goalname=max(self.spares,key=lambda name:self.pos(s,name)[0]);goal=self.pos(s,self.goalname)
        q=np.array([0,.21,0,0,0,-.21,0]);tip=fk(np.r_[0,0,0,q])[0]
        end=np.r_[goal[:2]+tip[:2]+np.array([.039,0]),math.pi,q]
        pre=end.copy();pre[0]+=.2
        self.stage=10;self.north=False
        v=p.copy()
        if p[0]>3.8 and p[1]>.75:v[1]=2.;self.add(v,1);v[0]=3.1;self.add(v,1)
        v=end.copy();v[0]=0;self.add(v,1);self.add(pre,1);self.add(end,-1)
    def nextstage(self,s,p,held):
        if self.stage==10:
            if not held:
                v=p.copy();v[0]-=.012;self.add(v,-1);return
            self.stage=3;v=p.copy();v[4]-=.2;v[0]+=.25;self.add(v,0);return
        if self.stage==0:
            if not held:
                self.attempt+=1;self.setup(s);return
            self.stage=1;self.add(self.high,0)
            v=self.high.copy();v[:2]+=np.array([-1.,.3] if self.north else [-.65,.65]);self.add(v,1)
        elif self.stage==1:
            self.cleared=True;self.stage=2;self.add(self.high,1);self.add(self.cp,1);self.add(self.cr,1);self.add(self.cg,-1)
        elif self.stage==2:
            if not held:
                self.attempt+=1;self.setup(s);return
            self.stage=3;self.add(self.high,0)
        elif self.stage==3:
            self.stage=4
            v=p.copy()
            if self.north:v[1]=2.;self.add(v,0)
            v[0]=3.1;self.add(v,0)
            v[2:]=np.r_[0,[0,.15,0,-.4,0,-.05,0]];self.add(v,0)
            v[1]=-.5;self.add(v,0)
        elif self.stage==4:
            self.stage=5;v=p.copy();v[:2]+=self.pos(s,'plate')[:2]-self.pos(s,self.goalname)[:2];self.add(v,1)
        else:self.add(p,1)
    def get_action(self,s):
        self.ticks+=1;p=np.array([s.get(self.r,f) for f in self.fs]);held=s.get(self.r,'grasp_active')>.5
        if self.prev is not None and self.lastmoving and np.max(abs(p-self.prev))<1e-5:self.stuck+=1
        else:self.stuck=0
        self.prev=p.copy()
        if self.stuck>=3 and self.stage in (1,3) and held:
            lift=p.copy();lift[4]=max(-.52,p[4]-.1);self.queue.insert(0,(lift,0));self.stuck=0
        if self.stuck>=3 and self.stage in (0,2) and not held:
            self.attempt+=1;self.setup(s);self.stuck=0
        if not self.queue:self.nextstage(s,p,held)
        target,grip=self.queue[0];d=target-p;d[[2,7,9]]=(d[[2,7,9]]+math.pi)%(2*math.pi)-math.pi
        a=np.zeros(11);a[:10]=d*min(1.,.2/max(1e-9,np.max(abs(d))))
        # Keep fingers open until the observed tool configuration has arrived.
        if np.max(abs(d))<1e-4:a[10]=grip
        elif grip:a[10]=1 if not held else 0
        if np.max(abs(d))<1e-4:self.queue.pop(0)
        self.lastmoving=np.max(abs(a[:10]))>1e-4
        return a.astype(np.float32)
