import numpy as np
import math
from motion import path_translation
from pose_motion import path_pose

def wrap(a):
    return (a+math.pi)%(2*math.pi)-math.pi

def unit(a):
    return np.array([math.cos(a),math.sin(a)])

def rot(v,a):
    c,s=math.cos(a),math.sin(a)
    return np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]])

class GeneratedApproach:
    """Grasp the shaft, plan a collision-free contact pose, and push along a face normal.

    All planning uses the observed rigid geometry. No simulator calls are needed.
    The lattice planners have bounded searches and are only run between pushes.
    """
    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low)
        self.high = np.asarray(action_space.high)
    def reset(self, state, info):
        s=np.asarray(state);self.phase=-2
        u=unit(s[11]);p=s[9:11]-1.04*u
        direction=p-s[:2];direction/=np.linalg.norm(direction)
        self.angle=math.atan2(direction[1],direction[0])
        self.base=p-.19*direction
        self.t=0;self.initial_hook=s[9:12].copy()
        self.previous=None;self.contact_speed=.005;self.stalled=0
        self.pose_attempted=False
    def plan_push(self,s):
        delta=s[29:31]-s[20:22]
        direction=delta/max(np.linalg.norm(delta),1e-8)
        ang=math.atan2(direction[1],direction[0])
        angles=list(np.linspace(.50,2.65,44))
        angles += [wrap(ang+k*math.pi/2) for k in range(-2,3)]
        best=None
        for theta in angles:
            if not .45<theta<2.7:continue
            u=unit(theta);v=rot(u,math.pi/2)
            offset=rot(s[9:11]-s[:2],wrap(theta-s[11]))
            for bar in [0,1]:
                normal=u if bar==0 else v
                normal=normal*(1 if np.dot(normal,direction)>=0 else -1)
                error=math.acos(np.clip(np.dot(normal,direction),-1,1))
                if error>1.15:continue
                choices=[.3,.2,.4,.12,.5] if bar==0 else [.16,.25,.4,.6,.8]
                tangent=v if bar==0 else u
                for along in choices:
                    corner=s[20:22]-.13*normal+along*tangent
                    base=corner-offset
                    ends=np.array([corner,corner-s[18]*u,corner-s[19]*v])
                    violation=max(0,.115-base[1],base[1]-1.125,.14-base[0],base[0]-3.36,.06-ends[:,1].min(),ends[:,1].max()-2.44,.06-ends[:,0].min(),ends[:,0].max()-3.44)
                    travel=max(0,float(np.dot(delta,normal))-.06)+.04
                    future=base+travel*normal
                    future_ends=ends+travel*normal
                    future_violation=max(0,.115-future[1],future[1]-1.125,.14-future[0],future[0]-3.36,.06-future_ends[:,1].min(),future_ends[:,1].max()-2.44,.06-future_ends[:,0].min(),future_ends[:,0].max()-3.44)
                    cost=violation*100+future_violation*10+error+(.02 if bar else 0)+abs(along-.3)*.01
                    if best is None or cost<best[0]:best=(cost,theta,corner,normal)
        _,self.theta,self.corner,self.normal=best
        self.side=1
    def get_action(self, state):
        s=np.asarray(state);self.t+=1
        still=self.previous is not None and np.linalg.norm(s[:5]-self.previous)<1e-6
        self.stalled=self.stalled+1 if still else 0
        if self.phase==0 and still:
            self.contact_speed=max(.0001,self.contact_speed*.5)
        self.previous=s[:5].copy()
        a=np.array([0.,0.,0.,0.,1.])
        if self.phase==-2:
            u=unit(s[11]); rel=s[:2]-s[9:11]
            near=s[9:11]+np.clip(np.dot(rel,-u),0,s[18])*(-u)
            away=s[:2]-near;dist=np.linalg.norm(away)
            a[4]=0;a[3]=.1-s[4]
            if dist<.30:
                destination=np.clip(s[:2]+away/max(dist,.001)*(.31-dist),[.13,.13],[3.37,1.10])
                a[:2]=destination-s[:2]
            else:
                v=rot(u,math.pi/2)
                direction=-v if np.dot(s[:2]-s[9:11],v)>0 else v
                length=1.08
                if u[1]>.1:
                    length=min(length,(s[10]-.37*direction[1]-.14)/u[1])
                point=s[9:11]-length*u
                self.stage=point-.37*direction
                self.angle=math.atan2(direction[1],direction[0]);self.base=point-.19*direction
                self.phase=-3
        elif self.phase==-3:
            a[4]=0;a[3]=.1-s[4];a[:2]=self.stage-s[:2]
            if np.linalg.norm(a[:2])<.005:self.phase=-1
        elif self.phase==-1:
            a[4]=0
            a[2]=wrap(self.angle-s[2]);a[3]=.1-s[4]
            if abs(a[2])<.005:self.phase=0
        elif self.phase==0:
            err=self.base-s[:2];da=wrap(self.angle-s[2])
            speed=.05 if np.linalg.norm(err)>.4 else self.contact_speed
            a[:2]=np.clip(err,-speed,speed);a[2]=da;a[3]=min(self.contact_speed,.2-s[4])
            if self.stalled>=4:
                a[:2]=-.012*unit(s[2]);a[3]=0
                self.contact_speed=.005
            if np.linalg.norm(s[9:12]-self.initial_hook)>.001:self.phase=1
        elif self.phase==1:
            if not self.pose_attempted:
                self.pose_attempted=True
                self.plan_push(s)
                offset=rot(s[9:11]-s[:2],wrap(self.theta-s[11]))
                self.pose_route=path_pose(s,np.r_[self.corner-offset,self.theta])
                if self.pose_route:
                    self.phase=8
                    return a.astype(np.float32)
            a[:2]=np.array([1.65,.70])-s[:2]
            if np.linalg.norm(a[:2])<.005:
                self.plan_push(s);self.phase=2
        elif self.phase==2:
            a[2]=wrap(self.theta-s[11])
            if abs(a[2])<.005:
                goal=self.corner-(s[9:11]-s[:2])
                self.route=path_translation(s,goal)
                self.phase=7 if self.route else 3
        elif self.phase==3:
            safe_x=max(.65,s[18]*math.cos(s[11])+.08,(s[9]-s[0])+.14)
            a[0]=safe_x-s[9]
            if abs(a[0])<.005:self.phase=4
        elif self.phase==4:
            a[1]=self.corner[1]-s[10]
            if abs(a[1])<.005:self.phase=5
        elif self.phase==5:
            a[0]=self.corner[0]-s[9]
            if abs(a[0])<.005:
                self.phase=6;self.push_start=s[20:22].copy();self.push_t=0
        elif self.phase==8:
            while self.pose_route and np.linalg.norm(self.pose_route[0][:2]-s[:2])<.004 and abs(wrap(self.pose_route[0][2]-s[11]))<.004:
                self.pose_route.pop(0)
            if self.pose_route:
                a[:2]=self.pose_route[0][:2]-s[:2]
                a[2]=wrap(self.pose_route[0][2]-s[11])
                if self.stalled>=5:self.phase=1
            else:self.phase=6;self.push_t=0
        elif self.phase==7:
            while self.route and np.linalg.norm(self.route[0]-s[:2])<.006:self.route.pop(0)
            if self.route:a[:2]=self.route[0]-s[:2]
            else:self.phase=6;self.push_t=0
        else:
            self.push_t+=1
            delta=s[29:31]-s[20:22]
            a[:2]=self.side*self.normal*.01
            if np.dot(delta,self.side*self.normal)<.01 or self.push_t>80 or self.stalled>=5:
                self.phase=1;self.pose_attempted=False
        return np.clip(a,self.low,self.high).astype(np.float32)
