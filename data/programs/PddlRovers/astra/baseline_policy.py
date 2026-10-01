import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
    def xy(self,s,o):
        return np.array([s.get(o,'x'),s.get(o,'y')])
    def reset(self,s,info):
        self.rovers=sorted(s.get_objects(self.space.get_type('rover')),key=lambda o:o.name)
        self.objs=list(s.get_objects(self.space.get_type('objective')))
        self.samples=list(s.get_objects(self.space.get_type('sample')))
        self.lander=list(s.get_objects(self.space.get_type('lander')))[0]
        self.home=[self.xy(s,r) for r in self.rovers]
        self.obs=[]
        for o in s.get_objects(self.space.get_type('obstacle')):
            self.obs.append((self.xy(s,o),np.array([s.get(o,'half_x'),s.get(o,'half_y')]),s.get(o,'z')+s.get(o,'half_z')))
        self.land=self.xy(s,self.lander)
        self.wall=(np.array([0.,0.]),np.array([.05,2.0]),1.)
        self.block=self.obs+[self.wall,(self.land,np.array([.35,.25]),.5)]
        axis=np.linspace(-2.24,2.24,57)
        self.spacing=axis[1]-axis[0]
        xx,yy=np.meshgrid(axis,axis)
        self.points=np.column_stack([xx.ravel(),yy.ravel()])
        self.free=self.safe(self.points)
        self.ids=np.full(len(self.points),-1,int);self.ids[self.free]=np.arange(np.sum(self.free))
        self.nodes=self.points[self.free]
        grid=self.ids.reshape(len(axis),len(axis)); rows=[];cols=[];weights=[]
        for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
            for y in range(len(axis)):
                for x in range(len(axis)):
                    y2,x2=y+dy,x+dx
                    if 0<=y2<len(axis) and 0<=x2<len(axis) and grid[y,x]>=0 and grid[y2,x2]>=0:
                        if dx and dy and (grid[y,x2]<0 or grid[y2,x]<0):continue
                        a,b=grid[y,x],grid[y2,x2]
                        rows.extend([a,b]);cols.extend([b,a]);weights.extend([self.spacing,self.spacing])
        self.graph=csr_matrix((weights,(rows,cols)),shape=(len(self.nodes),len(self.nodes)))
        self.fields={};self.masks={};self.goal=[None,None];self.prev=None;self.fail={};self.t=0
        for j,o in enumerate(self.objs):self.masks[('image',j)]=self.visible(self.nodes,self.xy(s,o),2.)
        for j,o in enumerate(self.samples):self.masks[('sample',j)]=np.linalg.norm(self.nodes-self.xy(s,o),axis=1)<.235
        self.masks[('send',0)]=self.visible(self.nodes,self.land,4.,True) | ((self.nodes[:,0]>.3)&(self.nodes[:,1]<-2.05))
        for i in range(2):self.masks[('home',i)]=np.linalg.norm(self.nodes-self.home[i],axis=1)<.1
    def safe(self,p):
        p=np.atleast_2d(p);ok=(np.abs(p)<=2.25+1e-7).all(axis=1)
        for c,h,z in self.block:
            ok &= ~((np.abs(p-c)<h+.255).all(axis=1))
        return ok
    def visible(self,p,q,limit,radio=False):
        p=np.atleast_2d(p);v=q-p;ok=np.linalg.norm(v,axis=1)<=limit-.02
        for c,h,z in self.obs+[self.wall]:
            if z<.15:continue
            # Objectives and antenna are above low mounds.
            lo=c-h-.005;hi=c+h+.005
            inv=np.divide(1.,v,out=np.full_like(v,1e12),where=np.abs(v)>1e-10)
            a=(lo-p)*inv;b=(hi-p)*inv
            enter=np.maximum(np.minimum(a,b).max(axis=1),0.)
            leave=np.minimum(np.maximum(a,b).min(axis=1),.9999)
            ok &= enter>leave
        return ok
    def field(self,key):
        if key not in self.fields:
            ids=np.where(self.masks[key])[0]
            self.fields[key]=dijkstra(self.graph,directed=False,indices=ids,min_only=True) if len(ids) else np.full(len(self.nodes),np.inf)
        return self.fields[key]
    def nearest(self,p):return int(np.argmin(np.sum((self.nodes-p)**2,axis=1)))
    def motion(self,p,key,other):
        f=self.field(key);dist=np.max(np.abs(self.nodes-p),axis=1)
        candidates=np.where((dist<=.201)&(np.linalg.norm(self.nodes-other,axis=1)>.52))[0]
        if not len(candidates):return np.zeros(2)
        # Prefer progress, then the shortest displacement to break flat ties.
        costs=f[candidates]+dist[candidates]*.95
        for j in candidates[np.argsort(costs)[:20]]:
            q=self.nodes[j]
            if self.safe(np.array([p+(q-p)*a for a in [.33,.66,1.]])).all():return q-p
        return np.zeros(2)
    def get_action(self,s):
        self.t+=1;p=[self.xy(s,r) for r in self.rovers]
        have=np.array([[s.get(o,'have_image_rover'+str(i))>.5 for o in self.objs] for i in range(2)])
        recv=np.array([s.get(o,'received_image')>.5 for o in self.objs]);done=recv|have.any(axis=0)
        ana=np.array([[s.get(o,'analyzed_rover'+str(i))>.5 for o in self.samples] for i in range(2)])
        sr=np.array([s.get(o,'received_analysis')>.5 for o in self.samples]);soil=np.array([s.get(o,'is_soil')>.5 for o in self.samples])
        need=[not np.any((sr|ana.any(axis=0))&(soil==k)) for k in [False,True]]
        if self.prev is not None:
            old,goals,ops,oldhave,oldana=self.prev
            for i in range(2):
                key=goals[i]
                if key and key[0] in ('image','send') and np.linalg.norm(p[i]-old[i])<.015:
                    failed=(ops[i]==-1/6 and np.array_equal(oldhave[i],have[i])) or (ops[i]==-.5 and s.get(self.rovers[i],'calibrated')<.5) or (ops[i]==.5 and key[0]=='send')
                    if failed:
                        self.fail[(i,key)]=self.fail.get((i,key),0)+1
                        if self.fail[(i,key)]>=2:
                            self.masks[key] &= np.linalg.norm(self.nodes-p[i],axis=1)>.22
                            self.fields.pop(key,None);self.fail[(i,key)]=0
        # Assign undone objectives by their side; both sides are reachable via wall ends.
        reserved=set();action=np.zeros(8,dtype=np.float32);ops=[];used=[]
        for i in range(2):
            r=self.rovers[i];idx=self.nearest(p[i]);key=self.goal[i]
            valid=False
            if key:
                typ,j=key
                valid=(typ=='image' and not done[j]) or (typ=='sample' and need[int(soil[j])])
            if not valid:
                options=[]
                for j,o in enumerate(self.objs):
                    k=('image',j)
                    if not done[j] and k not in reserved:
                        own=0 if s.get(o,'x')>=0 else 1
                        options.append((self.field(k)[idx]+(0 if own==i else 4),k))
                for j,o in enumerate(self.samples):
                    k=('sample',j)
                    if need[int(soil[j])] and ('soil',int(soil[j])) not in reserved and s.get(r,'store_full')<.5:
                        options.append((self.field(k)[idx]+.15,k))
                options=[v for v in options if np.isfinite(v[0])]
                if options:key=min(options,key=lambda z:z[0])[1]
                else:
                    pending=np.any(have[i]&~recv) or np.any(ana[i]&~sr)
                    key=('send',0) if pending else ('home',i)
            self.goal[i]=key;reserved.add(key)
            if key[0]=='sample':reserved.add(('soil',int(soil[key[1]])))
            move=self.motion(p[i],key,p[1-i]);q=p[i]+move
            if key[0]=='home' and np.max(np.abs(self.home[i]-p[i]))<=.2 and np.linalg.norm(self.home[i]-p[1-i])>.52:
                move=self.home[i]-p[i];q=self.home[i]
            op=0.
            # Operate en route whenever useful; goal-directed camera pairs take priority.
            visible=self.visible(q,np.array([0,0]),0) if False else [bool(self.visible(q,self.xy(s,o),2.)[0]) for o in self.objs]
            imageable=[j for j in range(len(self.objs)) if visible[j] and not done[j]]
            if s.get(r,'store_full')>.5:op=5/6
            elif key[0]=='sample' and np.linalg.norm(q-self.xy(s,self.samples[key[1]]))<.245:op=-5/6
            elif imageable:op=-1/6 if s.get(r,'calibrated')>.5 else -.5
            elif (np.any(have[i]&~recv) or np.any(ana[i]&~sr)) and self.visible(q,self.land,4.,True)[0]:op=.5
            elif key[0]=='image' and self.field(key)[self.nearest(q)]<.05:op=-1/6 if s.get(r,'calibrated')>.5 else -.5
            elif key[0]=='send' and self.field(key)[self.nearest(q)]<.05:op=.5
            action[4*i:4*i+2]=move;action[4*i+3]=op;ops.append(op);used.append(key)
        self.prev=([v.copy() for v in p],used,ops,have.copy(),ana.copy())
        return action
