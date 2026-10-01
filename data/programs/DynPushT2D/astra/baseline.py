import numpy as np
import math
import heapq


def rot(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s],[s,c]])
def limit(a):
    return a*min(1.,.049/(np.max(np.abs(a))+1e-12))
def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass
    def reset(self,state,info):
        self.mode='choose';self.contact=None;self.path=[];self.count=0
        self.last=None;self.age=0
    def geometry(self,s):
        w,h,v=s[12:15]
        self.rects=[(-h/2,h/2,-w,0),(-w/2,w/2,-v-w,-w)]
        qs=[];ns=[]
        def edge(a,b,n,k):
            for t in np.linspace(.07,.93,k):qs.append(np.array(a)*(1-t)+np.array(b)*t);ns.append(n)
        edge((-h/2,0),(h/2,0),(0,1),19)
        edge((-h/2,-w),(-w/2,-w),(0,-1),8)
        edge((w/2,-w),(h/2,-w),(0,-1),8)
        edge((-h/2,-w),(-h/2,0),(-1,0),3)
        edge((h/2,-w),(h/2,0),(1,0),3)
        edge((-w/2,-v-w),(-w/2,-w),(-1,0),19)
        edge((w/2,-v-w),(w/2,-w),(1,0),19)
        edge((-w/2,-v-w),(w/2,-v-w),(0,-1),5)
        self.q=np.array(qs);self.n=np.array(ns)
        self.com=np.array([0.,-(h*w/2+v*(w+v/2))/(h+v)])
        cy=self.com[1]
        inertia=(h*((h*h+w*w)/12+(-w/2-cy)**2)+v*((v*v+w*w)/12+(-w-v/2-cy)**2))/(h+v)
        self.k=1/inertia
    def clear(self,pts,s,margin=.014):
        p=(np.asarray(pts)-s[:2])@rot(s[2]);ds=[]
        for x0,x1,y0,y1 in self.rects:
            dx=np.maximum(np.maximum(x0-p[...,0],p[...,0]-x1),0)
            dy=np.maximum(np.maximum(y0-p[...,1],p[...,1]-y1),0)
            ds.append(dx*dx+dy*dy)
        return np.minimum(*ds)>(s[28]+margin)**2
    def line(self,a,b,s):
        t=np.linspace(0,1,max(3,int(np.linalg.norm(b-a)/.035)+1))
        return bool(np.all(self.clear(a+(b-a)*t[:,None],s)))
    def route(self,start,goal,s):
        if self.line(start,goal,s):return [goal]
        step=.0824;N=59
        xs=np.arange(N)*step+.11
        xx,yy=np.meshgrid(xs,xs,indexing='ij');points=np.stack([xx,yy],axis=-1)
        free=self.clear(points,s,.018)
        a=tuple(np.clip(np.round((start-.11)/step).astype(int),0,N-1));b=tuple(np.clip(np.round((goal-.11)/step).astype(int),0,N-1))
        free[a]=True;free[b]=True
        heap=[(0,0,a)];cost={a:0};prev={};end=None
        while heap:
            _,g,u=heapq.heappop(heap)
            if g>cost[u]+1e-8:continue
            if u==b:end=u;break
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                z=(u[0]+di,u[1]+dj)
                if not(0<=z[0]<N and 0<=z[1]<N) or not free[z]:continue
                if di and dj and (not free[u[0]+di,u[1]] or not free[u[0],u[1]+dj]):continue
                ng=g+math.hypot(di,dj)
                if ng<cost.get(z,1e9):
                    cost[z]=ng;prev[z]=u;heapq.heappush(heap,(ng+math.hypot(z[0]-b[0],z[1]-b[1]),ng,z))
        if end is None:return None
        raw=[goal];u=b
        while u!=a:raw.append(points[u]);u=prev[u]
        raw.reverse();out=[];p=start;j=0
        while j<len(raw):
            k=j
            while k+1<len(raw) and self.line(p,raw[k+1],s):k+=1
            out.append(raw[k]);p=raw[k];j=k+1
        return out
    def scores(self,s):
        R=rot(s[2]);f=-self.n@R.T
        arm=(self.q-self.com)@R.T
        tor=arm[:,0]*f[:,1]-arm[:,1]*f[:,0]
        om=self.k*tor/(1+self.k*np.sum(arm*arm,axis=1))
        f=f-om[:,None]*np.column_stack([-arm[:,1],arm[:,0]])
        off=-self.com@R.T
        vel=f+om[:,None]*np.array([-off[1],off[0]])
        e=s[29:31]-s[:2];a=wrap(s[31]-s[2])
        # Desired gradient for the reference pose.
        angweight=.45
        tw=np.column_stack([vel,om*angweight]);want=np.r_[e,a*angweight]
        score=tw@want/(np.linalg.norm(tw,axis=1)+1e-9)
        return score,vel,om
    def get_action(self,state):
        a=self._action(state)
        p=np.asarray(state)[16:18]
        return np.clip(p+a,.10001,4.89999)-p
    def _action(self,state):
        s=np.asarray(state);self.count+=1
        if self.count==1:self.geometry(s)
        p=s[16:18];R=rot(s[2]);score,vel,om=self.scores(s)
        if self.mode=='choose':
            # Leave contact before planning a route around the inflated shape.
            if not self.clear(p,s,.04):
                pl=(p-s[:2])@R
                ds=[]
                for x0,x1,y0,y1 in self.rects:
                    nearest=np.clip(pl,[x0,y0],[x1,y1]);d=pl-nearest
                    ds.append(d)
                d=min(ds,key=lambda d:np.linalg.norm(d))
                if np.linalg.norm(d)<1e-6:d=R.T@(p-s[:2]);d[1]=abs(d[1])+.1
                return np.clip(R@d/(np.linalg.norm(d)+1e-9)*.045,-.049,.049)
            target=s[:2]+(self.q+self.n*(s[28]+.055))@R.T
            dist=np.linalg.norm(target-p,axis=1)
            merit=score/(1+.25*dist)
            valid=np.all((target>.13)&(target<4.87),axis=1)
            contactcenters=s[:2]+(self.q+self.n*(s[28]+.020))@R.T
            valid &= self.clear(contactcenters,s,.012)
            valid &= self.clear(target,s,.025)
            merit[~valid]=-1e9
            self.path=None
            for j in np.argsort(merit)[::-1][:12]:
                if not valid[j]:continue
                self.path=self.route(p,target[j],s)
                if self.path is not None:break
            if self.path is None:return np.zeros(2)
            self.contact=int(j);self.mode='approach';self.age=0
            self.planpose=s[:3].copy()
        j=self.contact;self.age+=1
        if self.mode=='approach':
            if np.linalg.norm(s[:2]-self.planpose[:2])>.012 or abs(wrap(s[2]-self.planpose[2]))>.025:
                self.mode='choose';return np.zeros(2)
            while len(self.path)>1 and np.linalg.norm(self.path[0]-p)<.005:self.path.pop(0)
            target=self.path[0]
            if len(self.path)==1 and np.linalg.norm(target-p)<.025:self.mode='push';self.age=0
            else:
                if self.age>180:self.mode='choose'
                action=limit(target-p)
                if not self.line(p,p+action,s):
                    self.mode='choose';return np.zeros(2)
                return action
        q=s[:2]+R@self.q[j];n=R@self.n[j]
        desired=q+n*s[28]
        if self.age>1 and (score[j]<max(score)*.70 or score[j]<.008 or self.age>65):
            self.mode='choose';return np.zeros(2)
        # Steer toward contact and press inward, reducing speed near completion.
        err=np.linalg.norm(s[29:31]-s[:2])+abs(wrap(s[31]-s[2]))*.4
        speed=min(.045,max(.004,err*.15))
        delta=desired-p
        action=delta-n*np.dot(delta,n)-n*(speed+max(0.,-np.dot(delta,n)))
        if np.linalg.norm(p-desired)>.22:self.mode='choose'
        return limit(action)
