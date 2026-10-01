from env_client import make_env
from selection_snapshot import GeneratedApproach
import numpy as np,time,sys

class SelectPolicy(GeneratedApproach):
    def choose(self,state):
        todo=[o for o in self.cubes if not self.done(state,o)]
        if not todo:self.obj=None;return
        pos={o.name:self.xyz(state,o) for o in todo}
        def key(o):
            p=pos[o.name]
            near=min((np.linalg.norm(p[:2]-v[:2]) for n,v in pos.items() if n!=o.name),default=1.)
            score=p[0] if MODE=='front' else near-.6*max(0,abs(p[1])-.08)
            return self.attempts[o.name],-score
        self.obj=min(todo,key=key)
        self.attempts[self.obj.name]+=1
        p=pos[self.obj.name]
        self.armreach=.55 if p[0]>-.065 else .65
        self.reach=self.armreach+.12
        self.src=np.r_[p[:2]+[self.reach,0],np.pi]
        self.pickz=np.clip(p[2]-.38,.025,.22)
        self.highz=max(.16,self.pickz+.10)
        self.base=self.src.copy();self.z=self.highz;self.grip=0;self.stage=0;self.age=0

MODE=sys.argv[1]
e=make_env();s,info=e.reset(seed=0,options={'object_count':20})
p=SelectPolicy(e.action_space,e.observation_space,{});p.reset(s,info)
bins=[s.get_object_from_name('bin_'+c) for c in ['red','green','blue','yellow']]
b0=np.array([p.xyz(s,o) for o in bins]);liftfail=0;liftpass=0;start=time.time()
for i in range(400):
    old=(p.obj.name if p.obj else None,p.stage)
    s,r,d,tr,info=e.step(p.get_action(s))
    new=(p.obj.name if p.obj else None,p.stage)
    if old[1]==3 and old!=new:
        if new[1]==4:liftpass+=1
        else:liftfail+=1
        print('LIFT',i,old,new,'pass',liftpass,'fail',liftfail,'sorted',sum(p.done(s,o) for o in p.cubes),flush=True)
    if d or tr:break
print('RESULT',MODE,'steps',i+1,'sorted',sum(p.done(s,o) for o in p.cubes),'pass',liftpass,'fail',liftfail,'bindelta',np.round(np.array([p.xyz(s,o) for o in bins])-b0,3).tolist(),'secs',time.time()-start,flush=True)
e.close()
