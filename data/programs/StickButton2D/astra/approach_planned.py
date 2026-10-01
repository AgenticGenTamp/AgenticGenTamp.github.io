import math
import numpy as np

TAU=2*math.pi

def wrap(a):
    return (a+math.pi)%TAU-math.pi

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.low=np.asarray(action_space.low)
        self.high=np.asarray(action_space.high)
    def reset(self,state,info):
        self.phase='direct'
        self.target=None
        self.ticks=0
        self.stalls=0
        self.prev=None
        self.goal=None
        self.held_geometry=None
        self.pick_first=self.choose_pick_first(state)
    def choose_pick_first(self,s):
        robot=next(iter(s.get_objects(self.space.get_type('crv_robot'))))
        r=self.read(s,robot,['x','y','theta','arm_joint'])
        stick=next(iter(s.get_objects(self.space.get_type('rectangle'))))
        st=self.read(s,stick,['x','y','theta','width','height'])
        buttons=[self.read(s,b,['x','y']) for b in s.get_objects(self.space.get_type('circle')) if b.name.startswith('button')]
        if not any(b[1]>1.34 for b in buttons):return False
        if len(buttons)>12:return True
        def steps(r,g):
            return max(np.max(abs(g[:2]-r[:2]))/.05,abs(wrap(g[2]-r[2]))/.196)
        def estimate(r,buttons,direct):
            r=r.copy();buttons=list(buttons);cost=0.
            if direct:
                while True:
                    goals=[(self.direct_goal(r,b),k) for k,b in enumerate(buttons) if b[1]<=1.34]
                    goals=[(steps(r,g),g,k) for g,k in goals if g is not None]
                    if not goals:break
                    c,g,k=min(goals,key=lambda a:a[0]);cost+=c;r=g;buttons.pop(k)
            sx,sy,_,w,h=st
            picks=[(max(abs(px-r[0])/.05,abs(wrap(t-r[2]))/.196),t,px) for t,px in [(math.pi,sx+w+.218),(0.,sx-.218)] if .101<px<3.399]
            c,t,px=min(picks)
            stage_y=max(.25,min(r[1],sy-.25))
            cost+=abs(r[1]-stage_y)/.05+c+abs(sy-.025-stage_y)/.05+2
            r=np.array([px,sy-.025,t,.2]);tool=st.copy()
            while buttons:
                goals=[(self.held_goal(r,tool,b),k) for k,b in enumerate(buttons)]
                goals=[(steps(r,g),g,k) for g,k in goals if g is not None]
                if not goals:return 1e5
                c,g,k=min(goals,key=lambda a:a[0]);cost+=c
                dt=wrap(g[2]-r[2]);co,si=math.cos(dt),math.sin(dt)
                tool[:2]=g[:2]+np.array([[co,-si],[si,co]])@(tool[:2]-r[:2]);tool[2]+=dt
                r=g;buttons.pop(k)
            return cost
        return estimate(r,buttons,False)<estimate(r,buttons,True)
    def read(self,s,o,fs):
        return np.array([s.get(o,f) for f in fs])
    def command(self,r,goal,vac=0):
        d=np.asarray(goal)-r
        d[2]=wrap(d[2])
        d=np.clip(d,self.low[:4],self.high[:4])
        # Keep both the base and rectangular gripper inside the workspace.
        for scale in (1.,.5,.25,.125,0.):
            dt=d[2]*scale;t=r[2]+dt;j=r[3]+d[3]
            c,ss=math.cos(t),math.sin(t)
            ext=np.array([.005*abs(c)+.035*abs(ss),.005*abs(ss)+.035*abs(c)])
            grip=j*np.array([c,ss])
            lo=np.maximum([.1001,.1001],ext-grip+.0001)
            hi=np.minimum([3.3999,1.1499],[3.4999-ext[0]-grip[0],1.1499])
            if self.held_geometry is not None:
                cd,sd=math.cos(dt),math.sin(dt)
                pts=self.held_geometry@np.array([[cd,sd],[-sd,cd]])
                lo=np.maximum(lo,-pts.min(axis=0)+.0001)
                hi=np.minimum(hi,np.array([3.4999,2.4999])-pts.max(axis=0))
            low=np.maximum(lo,r[:2]-.05);high=np.minimum(hi,r[:2]+.05)
            if np.all(low<=high):
                p=np.clip(r[:2]+d[:2],low,high)
                return np.array([p[0]-r[0],p[1]-r[1],dt,d[3],vac],dtype=np.float32)
        return np.array([0,0,0,0,vac],dtype=np.float32)
    def direct_goal(self,r,b):
        t=np.tile(np.linspace(-math.pi,math.pi,97),3)
        reach=np.repeat([.16,.20,.24],97)
        p=b-reach[:,None]*np.stack([np.cos(t),np.sin(t)],axis=1)
        ok=(p[:,0]>=.101)&(p[:,0]<=3.399)&(p[:,1]>=.101)&(p[:,1]<=1.149)
        dt=(t-r[2]+math.pi)%TAU-math.pi
        cost=np.maximum(np.max(abs(p-r[:2]),axis=1)/.05,abs(dt)/.196)
        cost[~ok]=np.inf
        k=np.argmin(cost)
        return None if not np.isfinite(cost[k]) else np.r_[p[k],t[k],.2]
    def held_goal(self,r,st,b):
        sx,sy,alpha,w,h=st
        u0=np.array([math.cos(alpha),math.sin(alpha)])
        v0=np.array([-math.sin(alpha),math.cos(alpha)])
        off=np.array([sx,sy])+w/2*u0-r[:2]
        dt=np.linspace(-math.pi,math.pi,145)
        c,ss=np.cos(dt),np.sin(dt)
        def rot(v):
            return np.stack([c*v[0]-ss*v[1],ss*v[0]+c*v[1]],axis=1)
        q=rot(off);v=rot(v0);u=rot(u0);t=r[2]+dt
        grip=.2*np.stack([np.cos(t),np.sin(t)],axis=1)
        ext=np.stack([.005*abs(np.cos(t))+.035*abs(np.sin(t)),.005*abs(np.sin(t))+.035*abs(np.cos(t))],axis=1)
        lo=np.maximum([.101,.101],ext-grip+.001)
        hi=np.minimum([3.399,1.149],np.stack([3.499-ext[:,0]-grip[:,0],np.full(len(t),1.149)],axis=1))
        corners=np.stack([q-w/2*u,q+w/2*u,q-w/2*u+h*v,q+w/2*u+h*v],axis=1)
        lo=np.maximum(lo,-corners.min(axis=1)+.001)
        hi=np.minimum(hi,np.array([3.499,2.499])-corners.max(axis=1))
        lmin=np.full(len(t),-.027);lmax=np.full(len(t),h+.027)
        for k in range(2):
            flat=abs(v[:,k])<1e-8
            safe=np.where(flat,1.,v[:,k])
            a=(b[k]-q[:,k]-hi[:,k])/safe;bb=(b[k]-q[:,k]-lo[:,k])/safe
            lmin=np.maximum(lmin,np.where(flat,-np.inf,np.minimum(a,bb)))
            lmax=np.minimum(lmax,np.where(flat,np.inf,np.maximum(a,bb)))
            lmax[flat&((b[k]-q[:,k]<lo[:,k])|(b[k]-q[:,k]>hi[:,k]))]=-100
        lengths=np.clip(np.sum((b-q-r[:2])*v,axis=1),lmin,np.maximum(lmin,lmax))
        p=b-q-lengths[:,None]*v
        cost=np.maximum(np.max(abs(p-r[:2]),axis=1)/.05,abs(dt)/.196)+.02*abs(dt)
        cost[(lmin>lmax)|np.any(lo>hi,axis=1)]=np.inf
        k=np.argmin(cost)
        return None if not np.isfinite(cost[k]) else np.r_[p[k],wrap(t[k]),.2]
    def get_action(self,s):
        self.ticks+=1
        robot=next(iter(s.get_objects(self.space.get_type('crv_robot'))))
        r=self.read(s,robot,['x','y','theta','arm_joint'])
        buttons=[o for o in s.get_objects(self.space.get_type('circle')) if o.name.startswith('button') and s.get(o,'color_g')<.5]
        if not buttons:return np.zeros(5,dtype=np.float32)
        sticks=list(s.get_objects(self.space.get_type('rectangle')))
        stick=next((o for o in sticks if o.name=='stick'),sticks[0] if sticks else None)
        st=self.read(s,stick,['x','y','theta','width','height']) if stick is not None else None
        if self.prev is not None and np.max(abs(r-self.prev))<1e-6:self.stalls+=1
        else:self.stalls=0
        self.prev=r.copy()
        if self.phase=='direct':
            candidates=[]
            for b in buttons:
                if s.get(b,'y')>1.34:continue
                g=self.direct_goal(r,self.read(s,b,['x','y']))
                if g is not None:candidates.append((np.max(abs(g[:2]-r[:2]))/.05+abs(wrap(g[2]-r[2]))/.196,b,g))
            if candidates and not self.pick_first:
                _,b,g=min(candidates,key=lambda a:a[0])
                if self.stalls<6:return self.command(r,g)
            if stick is None:return self.command(r,self.direct_goal(r,self.read(s,buttons[0],['x','y'])))
            self.phase='pickup'
            self.pick_st=st.copy()
            sx,sy,_,w,h=st
            picks=[]
            for theta,px in [(math.pi,sx+w+.218),(0.,sx-.218)]:
                if .101<px<3.399:
                    cost=max(abs(px-r[0])/.05,abs(wrap(theta-r[2]))/.196)
                    picks.append((cost,theta,px))
            _,self.pick_theta,px=min(picks)
            self.pick_goal=np.array([px,sy-.025,self.pick_theta,.2])
            self.stage_y=min(r[1],sy-.25)
            self.stage_y=max(.25,self.stage_y)
            self.phase='retreat'
            self.retreat_goal=r.copy();self.retreat_goal[1]=self.stage_y
        if self.phase=='retreat':
            if abs(r[1]-self.stage_y)<.005:self.phase='stage'
            else:return self.command(r,self.retreat_goal)
        if self.phase=='stage':
            g=self.pick_goal.copy();g[1]=self.stage_y
            if np.max(abs(g[:2]-r[:2]))<.005 and abs(wrap(g[2]-r[2]))<.01:self.phase='pickup'
            else:return self.command(r,g)
        if self.phase=='pickup':
            if np.max(abs(st[:2]-self.pick_st[:2]))>.002:
                self.phase='held';self.target=None;self.goal=None
            else:
                g=self.pick_goal.copy()
                if np.max(abs(g[:2]-r[:2]))<.005:
                    g[0]+=.04 if self.pick_theta==math.pi else -.04
                return self.command(r,g,1)
        if self.phase=='held':
            sx,sy,alpha,w,h=st
            u=np.array([math.cos(alpha),math.sin(alpha)])
            v=np.array([-math.sin(alpha),math.cos(alpha)])
            self.held_geometry=np.array([np.array([sx,sy])+a*w*u+b*h*v-r[:2] for a,b in [(0,0),(1,0),(0,1),(1,1)]])
            if self.target not in [b.name for b in buttons] or self.goal is None or self.stalls>4:
                goals=[]
                for b in buttons:
                    g=self.held_goal(r,st,self.read(s,b,['x','y']))
                    if g is not None:
                        cost=max(np.max(abs(g[:2]-r[:2]))/.05,abs(wrap(g[2]-r[2]))/.196)
                        goals.append((cost,b.name,g))
                if goals:
                    _,self.target,self.goal=min(goals,key=lambda a:a[0])
            if self.goal is not None:return self.command(r,self.goal,1)
        return np.array([0,0,0,0,1],dtype=np.float32)
