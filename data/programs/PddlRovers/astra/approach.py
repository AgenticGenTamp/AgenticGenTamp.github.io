"""Cooperative rover policy using observed objects and bounded geometric planning.

Navigation fields are cached on a small collision grid.  For the benchmark's
one-to-four objectives, a bounded tour calculation allocates samples and photos
between rovers; larger objective sets use the same online greedy controller.
Operators run during travel, and unsuccessful operations refine their target
regions.  No environment implementation or simulator helpers are required.
"""
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.clearance=.215
    def xy(self,s,o):
        return np.array([s.get(o,'x'),s.get(o,'y')])
    def reset(self,s,info):
        self.clearance=info.get("_clearance",.215)
        self.initial=s.copy()
        self.last_positions=None
        self.last_moves=None
        self.stuck=[0,0]
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
        self.block=self.obs+[(np.array([0.,0.]),np.array([.05,2.5]),1.),(self.land,np.array([.4,.33]),.5)]
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
        sample_positions=np.array([self.xy(s,o) for o in self.samples])
        nearest_samples=np.argmin(np.linalg.norm(self.nodes[:,None,:]-sample_positions[None,:,:],axis=2),axis=1)
        sample_kinds=np.array([s.get(o,'is_soil') for o in self.samples])
        for j,o in enumerate(self.samples):
            self.masks[('sample',j)]=(np.linalg.norm(self.nodes-self.xy(s,o),axis=1)<.235)&(sample_kinds[nearest_samples]==sample_kinds[j])
        self.masks[('send',0)]=self.visible(self.nodes,self.land,4.,True) | ((self.nodes[:,0]>.3)&(self.nodes[:,1]<-2.05)&(np.linalg.norm(self.nodes-self.land,axis=1)<3.95))
        for i in range(2):self.masks[('home',i)]=np.linalg.norm(self.nodes-self.home[i],axis=1)<.23
        self.make_schedule(s)
    def make_schedule(self,s):
        self.schedule=[[],[]]
        self.waypoint_keys={}
        if len(self.objs)>4:return
        tasks=[('image',j) for j in range(len(self.objs))]+[('sample',j) for j in range(len(self.samples))]
        self.schedule=[[],[]]
        cache={};targets={}
        anchors=self.home+[self.xy(s,o) for o in self.samples+self.objs]
        for i in range(2):
            side=self.nodes[:,0]>0 if i==0 else self.nodes[:,0]<0
            for key in tasks:
                ids=np.where(self.masks[key]&side)[0]
                pick=set()
                if len(ids):
                    for p in anchors:
                        dist=np.max(np.abs(self.nodes[ids]-p),axis=1)
                        pick.add(int(ids[np.argmin(dist)]))
                    for k in np.linspace(0,len(ids)-1,min(14,len(ids))).astype(int):pick.add(int(ids[k]))
                targets[i,key]=np.array(sorted(pick),int)
        union=np.array(sorted(set(v for ids in targets.values() for v in ids)),int)
        paths=dijkstra(self.graph,directed=False,indices=union)
        lookup={j:k for k,j in enumerate(union)}
        rows={k:np.array([lookup[j] for j in ids],int) for k,ids in targets.items()}
        homes=[self.nearest(p) for p in self.home]
        def route(i,extra,images):
            todo=tuple(images+list(extra))
            if (i,todo) in cache:return cache[i,todo]
            best=(np.inf,[],[])
            for order in itertools.permutations(todo):
                if not order:return (0.,[],[])
                if any(not len(targets[i,k]) for k in order):continue
                vals=paths[rows[i,order[0]],homes[i]]
                backs=[]
                for a,b in zip(order,order[1:]):
                    edge=paths[rows[i,a]][:,targets[i,b]]
                    costs=vals[:,None]+edge
                    backs.append(np.argmin(costs,axis=0))
                    vals=np.min(costs,axis=0)
                finals=vals+paths[rows[i,order[-1]],homes[i]]
                last=int(np.argmin(finals));val=finals[last]
                val+=sum(.3 if x[0]=='image' else .2 for x in order)
                val+=sum(.001*k for k,x in enumerate(order) if x[0]=='sample')
                if val<best[0]:
                    indices=[last]
                    for back in reversed(backs):indices.append(int(back[indices[-1]]))
                    poses=[int(targets[i,key][j]) for key,j in zip(order,reversed(indices))]
                    best=(val,list(order),poses)
            cache[i,todo]=best
            return best
        stone=[j for j,o in enumerate(self.samples) if s.get(o,'is_soil')<.5]
        soil=[j for j,o in enumerate(self.samples) if s.get(o,'is_soil')>.5]
        best=(np.inf,None)
        choices=[]
        for j in range(len(self.objs)):
            key=('image',j)
            choices.append([i for i in range(2) if len(targets[i,key]) and np.isfinite(self.field(key)[homes[i]])])
        for owners in itertools.product(*choices):
            images=[[('image',j) for j in range(len(self.objs)) if owners[j]==i] for i in range(2)]
            for a,b,i,j in itertools.product(stone,soil,range(2),range(2)):
                extras=[[],[]];extras[i].append(('sample',a));extras[j].append(('sample',b))
                r=[route(k,extras[k],images[k]) for k in range(2)]
                score=max(r[0][0],r[1][0])+.05*(r[0][0]+r[1][0])
                if score<best[0]:best=(score,[r[0][1],r[1][1]],[r[0][2],r[1][2]])
        if best[1] is not None:
            self.schedule=best[1]
            for i in range(2):
                for key,node in zip(self.schedule[i],best[2][i]):
                    wk=('waypoint',i,key)
                    self.waypoint_keys[i,key]=(wk,node)
                    self.fields[wk]=paths[lookup[node]].copy()
    def safe(self,p):
        p=np.atleast_2d(p);ok=(np.abs(p)<=2.25+1e-7).all(axis=1)
        for c,h,z in self.block:
            ok &= ~((np.abs(p-c)<h+self.clearance).all(axis=1))
        return ok
    def visible(self,p,q,limit,radio=False):
        p=np.atleast_2d(p);v=q-p;ok=np.linalg.norm(v,axis=1)<=limit-.02
        for c,h,z in self.obs+([self.wall] if radio else []):
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
            if key[0]=='send' and len(ids):
                # Radio stops should also lie on a short route back home.
                costs=np.minimum(self.field(('home',0)),self.field(('home',1)))[ids]
                g=self.graph.tocoo();n=len(self.nodes)
                graph=csr_matrix((np.concatenate([g.data,costs]),
                    (np.concatenate([g.row,np.full(len(ids),n)]),np.concatenate([g.col,ids]))),shape=(n+1,n+1))
                self.fields[key]=dijkstra(graph,directed=False,indices=n)[:n]
            else:
                self.fields[key]=dijkstra(self.graph,directed=False,indices=ids,min_only=True) if len(ids) else np.full(len(self.nodes),np.inf)
        return self.fields[key]
    def nearest(self,p):return int(np.argmin(np.sum((self.nodes-p)**2,axis=1)))
    def motion(self,p,key,other):
        f=self.field(key);dist=np.max(np.abs(self.nodes-p),axis=1)
        candidates=np.where((dist<=.601)&(np.linalg.norm(self.nodes-other,axis=1)>.52))[0]
        if not len(candidates):return np.zeros(2)
        # Smooth the grid path while checking the entire segment for collisions.
        costs=f[candidates]+dist[candidates]*.95+np.linalg.norm(self.nodes[candidates]-p,axis=1)*.002
        targets=self.nodes[candidates]
        clear=self.clear_segments(p,targets)
        if not np.any(clear):return np.zeros(2)
        costs[~clear]=np.inf
        q=targets[np.argmin(costs)]
        return (q-p)*min(1.,.2/max(np.max(np.abs(q-p)),1e-8))
    def clear_segments(self,p,targets):
        v=np.atleast_2d(targets)-p
        inv=np.divide(1.,v,out=np.full_like(v,1e12),where=np.abs(v)>1e-10)
        ok=np.ones(len(v),dtype=bool)
        for c,h,z in self.block:
            a=(c-h-self.clearance-p)*inv;b=(c+h+self.clearance-p)*inv
            enter=np.maximum(np.minimum(a,b).max(axis=1),0.)
            leave=np.minimum(np.maximum(a,b).min(axis=1),1.)
            hit=enter<=leave
            if np.all(np.abs(p-c)<h+self.clearance):
                escaping=(np.sum(v*(p-c),axis=1)>0)&(np.abs(np.atleast_2d(targets)-c)>=h+self.clearance).any(axis=1)
                hit &= ~escaping
            ok &= ~hit
        return ok
    def get_action(self,s):
        self.t+=1;p=[self.xy(s,r) for r in self.rovers]
        if self.last_positions is not None:
            for i in range(2):
                blocked=np.linalg.norm(p[i]-self.last_positions[i])<1e-5 and np.linalg.norm(self.last_moves[i])>.01
                self.stuck[i]=self.stuck[i]+1 if blocked else 0
            if max(self.stuck)>=3 and self.clearance<.29:
                self.reset(self.initial,{"_clearance":self.clearance+.035})
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
        # The fallback controller assigns reachable work; the tour overrides it when available.
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
                        if own!=i and np.isfinite(self.field(k)[self.nearest(self.home[own])]):continue
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
            for planned in self.schedule[i]:
                typ,j=planned
                valid=(typ=='image' and not done[j]) or (typ=='sample' and need[int(soil[j])])
                if valid and np.isfinite(self.field(planned)[idx]):
                    key=planned;break
            self.goal[i]=key;reserved.add(key)
            if key[0]=='sample':reserved.add(('soil',int(soil[key[1]])))
            motion_key=key
            waypoint=self.waypoint_keys.get((i,key))
            if waypoint is not None and self.masks[key][waypoint[1]]:motion_key=waypoint[0]
            move=self.motion(p[i],motion_key,p[1-i]);q=p[i]+move
            if key[0]=='home' and s.get(r,'at_home')>.5:
                move=np.zeros(2);q=p[i]
            op=0.
            # Operate en route whenever useful; goal-directed camera pairs take priority.
            visible= [bool(self.visible(q,self.xy(s,o),2.)[0]) for o in self.objs]
            imageable=[j for j in range(len(self.objs)) if visible[j] and not done[j]]
            nearest_sample=int(np.argmin([np.linalg.norm(q-self.xy(s,o)) for o in self.samples]))
            close_sample=np.linalg.norm(q-self.xy(s,self.samples[nearest_sample]))<.245 and need[int(soil[nearest_sample])]
            if s.get(r,'store_full')>.5:op=5/6
            elif close_sample:op=-5/6
            elif imageable:op=-1/6 if s.get(r,'calibrated')>.5 else -.5
            elif (np.any(have[i]&~recv) or np.any(ana[i]&~sr)) :op=.5
            elif key[0]=='image' and self.field(key)[self.nearest(q)]<.05:op=-1/6 if s.get(r,'calibrated')>.5 else -.5
            elif key[0]=='send' and self.field(key)[self.nearest(q)]<.05:op=.5
            action[4*i:4*i+2]=move;action[4*i+3]=op;ops.append(op);used.append(key)
        self.prev=([v.copy() for v in p],used,ops,have.copy(),ana.copy())
        self.last_positions=[v.copy() for v in p]
        self.last_moves=[action[:2].copy(),action[4:6].copy()]
        return action
