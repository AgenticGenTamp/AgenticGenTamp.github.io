import numpy as np
from fast_ik import vertical_ik
from kinematics import fk

class GeneratedApproach:
    def __init__(self,action_space,observation_space,primitives):
        self.action_space=action_space;self.observation_space=observation_space
        self.home=np.array([0.,-.35,-np.pi,-2.5,0.,-.87,np.pi/2])
        self.cache={}
    def reset(self,s,info):
        self.robot=next(iter(s.get_objects(self.observation_space.get_type('Kinematic3DRobot'))))
        self.objects=list(s.get_objects(self.observation_space.get_type('Kinematic3DCuboid')))
        self.block=next(o for o in self.objects if o.name=='target_block')
        self.region=next(o for o in self.objects if o.name=='target_region')
        self.done=set();self.index=0;self.stage='select';self.goal=None;self.wait=0;self.prev=None;self.stuck=0;self.retry=0
    def pos(self,s,o):return np.array([s.get(o,'pose_'+v) for v in 'xyz'])
    def robotvec(self,s):return np.array([s.get(self.robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]])
    def ik(self,z,x=.5):
        return vertical_ik(z,x)
    def parking(self,s,p):
        others=[o for o in self.objects if o!=self.obj]
        best=None;score=-1
        for x in [.15,.24,.33,.42]:
            for y in [-.30,-.15,0,.15,.30]:
                cand=np.array([x,y,.075+s.get(self.obj,'half_extent_z')+.0001])
                clearance=min(np.linalg.norm((cand[:2]-self.pos(s,o)[:2])/[1.,1.2])-max(s.get(o,'half_extent_x'),s.get(o,'half_extent_y')) for o in others)
                if clearance>score:best=cand;score=clearance
        return best
    def goto(self,stage,goal):self.stage=stage;self.goal=np.array(goal);self.wait=0;self.stuck=0
    def get_action(self,s):
        a=np.zeros(11,dtype=np.float32);cur=self.robotvec(s)
        held=next((o for o in self.objects if s.get(o,'grasp_active')>.5),None)
        if self.prev is not None and self.goal is not None:
            self.stuck=self.stuck+1 if np.max(abs(cur-self.prev))<1e-6 else 0
        self.prev=cur.copy();self.wait+=1
        if self.stage=='retract' and (np.max(abs(self.goal-cur))<.001 or self.stuck>3):self.stage='select'
        if self.stage=='select':
            obs=[o for o in self.objects if o.name.startswith('obstruction') and o.name not in self.done]
            self.obj=min(obs,key=lambda o:(-self.pos(s,o)[2]-s.get(o,'half_extent_z'),self.pos(s,o)[1])) if obs else self.block
            p=self.pos(s,self.obj);self.pick=p.copy()
            offsets=[(0,0),(.015,0),(-.015,0),(0,-.015),(0,.015),(.03,0),(-.03,0)]
            dx,dy=offsets[self.retry%len(offsets)]
            self.base=np.array([min(-.17,p[0]-.5+dx),p[1]-.001348925+dy,0]);self.reach=p[0]+dx-self.base[0]
            self.depth=p[2]+.08
            self.goto('hover',np.r_[self.base,self.ik(.30,self.reach)])
        if self.stage=='hover' and np.max(abs(self.goal-cur))<.001:
            self.goto('descend',np.r_[self.base,self.ik(self.depth,self.reach)])
        if self.stage=='descend':
            a[10]=-1
            if held is not None:
                self.obj=held;self.pick=self.pos(s,held)
                T=fk(cur[3:]);tip=T[:3,3]+T[:3,:3]@[0,0,-.181525034]+[.119899783,.000348925,-.005199842];self.proxy=tip[2];self.reach=tip[0];self.xoffset=cur[0]+tip[0]-self.pick[0]
                self.offset=self.proxy-self.pick[2]
                self.goto('lift',np.r_[cur[:3],self.ik(min(.38,self.proxy+.10),self.reach)])
            elif np.max(abs(self.goal-cur))<.001:
                self.depth-=.004
                self.goto('descend',np.r_[self.base,self.ik(self.depth,self.reach)])
            if self.stuck>2 or self.depth<self.pick[2]+.01:
                self.retry+=1;self.goto('retry',np.r_[cur[:3],self.ik(.30,self.reach)])
        if self.stage=='retry' and (np.max(abs(self.goal-cur))<.001 or self.stuck>3):self.stage='select';return a
        if self.stage=='lift' and (np.max(abs(self.goal-cur))<.001 or self.stuck>2):
            p=self.pos(s,self.obj)
            self.dest=self.pos(s,self.region).copy() if self.obj==self.block else self.parking(s,p)
            if self.obj==self.block:self.dest[2]+=s.get(self.region,'half_extent_z')+s.get(self.obj,'half_extent_z')+.0001
            goal=cur.copy();goal[:2]+=self.dest[:2]-p[:2]
            if goal[0]>-.17:
                self.reach+=goal[0]+.17;goal[0]=-.17
                goal[3:]=self.ik(self.proxy+.10,self.reach)
            self.goto('transport',goal)
        if self.stage=='transport' and np.max(abs(self.goal-cur))<.001:
            self.goto('place',np.r_[cur[:3],self.ik(self.dest[2]+self.offset,self.reach)])
        if self.stage=='place' and np.max(abs(self.goal-cur))<.001:self.stage='release'
        if self.stage=='release':
            a[10]=1
            if held is None:
                self.done.add(self.obj.name);self.index+=1;self.retry=0;self.goto('retract',np.r_[cur[:3],self.ik(.30,self.reach)])
            return a
        if self.goal is not None:
            d=self.goal-cur
            if self.stage in ['lift','place','descend','retract'] and max(abs(d[:3]))<.001:
                T=fk(cur[3:]);tip=T[:3,3]+T[:3,:3]@[0,0,-.181525034]+[.119899783,.000348925,-.005199842]
                G=fk(self.goal[3:]);end=G[:3,3]+G[:3,:3]@[0,0,-.181525034]+[.119899783,.000348925,-.005199842]
                v=end-tip;way=tip+v*min(1,(.03 if tip[2]>.22 else .010)/max(np.linalg.norm(v),1e-8))
                d[3:]=self.ik(way[2],way[0])-cur[3:]
            a[:10]=d*min(1,.12/max(np.max(abs(d)),1e-8))
        if self.stage=='descend':a[10]=-1
        return np.clip(a,self.action_space.low,self.action_space.high)
