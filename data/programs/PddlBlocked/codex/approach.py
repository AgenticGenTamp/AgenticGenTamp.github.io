"""Fast geometric manipulation policy for PR2Blocked."""
import math
import numpy as np
from scipy.optimize import least_squares

class GeneratedApproach:
    Q=np.array([.16752,.21063,1.4,-.24177,-2.99431,-.83787,-3.14159])
    OFF=np.array([.94433737,.34963603])
    TOOL_YAW=.7758
    def __init__(self,action_space,observation_space,primitives): self.dtype=action_space.dtype
    @staticmethod
    def w(x): return (x+math.pi)%(2*math.pi)-math.pi
    @staticmethod
    def _rot(axis,a):
        axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
        x,y,z=axis;c=math.cos(a);s=math.sin(a);C=1-c
        r=np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],
                    [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
                    [z*x*C-y*s,z*y*C+x*s,c+z*z*C]])
        t=np.eye(4);t[:3,:3]=r;return t
    @staticmethod
    def _trans(x,y,z):
        t=np.eye(4);t[:3,3]=[x,y,z];return t
    @classmethod
    def _fk(cls,base,q):
        t=cls._trans(base[0],base[1],0)@cls._rot([0,0,1],base[2])@cls._trans(-.05,.188,1.04)
        t=t@cls._rot([0,0,1],q[0])@cls._trans(.1,0,0)@cls._rot([0,1,0],q[1])@cls._rot([1,0,0],q[2])
        t=t@cls._trans(.4,0,0)@cls._rot([0,1,0],q[3])@cls._rot([1,0,0],q[4])
        return t@cls._trans(.321,0,0)@cls._rot([0,1,0],q[5])@cls._rot([1,0,0],q[6])@cls._trans(.18,0,0)
    def _tangent_ik(self,standard):
        """Nearby redundant pose, with the base 3 cm inside its x limit."""
        alternate=standard.copy();alternate[0]=4.97
        desired=self._fk(standard,self.Q)
        lo=np.array([-.715,-.524,-.8,-2.321,-math.pi,-2.094,-math.pi])
        hi=np.array([2.285,1.396,3.9,0,math.pi,0,math.pi])
        def residual(q):
            t=self._fk(alternate,q);e=t[:3,:3].T@desired[:3,:3]
            rv=np.array([e[2,1]-e[1,2],e[0,2]-e[2,0],e[1,0]-e[0,1]])*.5
            return np.r_[30*(t[:3,3]-desired[:3,3]),8*rv,.001*(q-self.Q)]
        q=least_squares(residual,self.Q,bounds=(lo,hi),max_nfev=120).x
        return alternate,q
    @staticmethod
    def g(s,n,f): return float(s.get(s.get_object_from_name(n),f))
    def reset(self,s,info):
        green=np.array([self.g(s,'green0','pose_x'),self.g(s,'green0','pose_y')])
        block=np.array([self.g(s,'blocker','pose_x'),self.g(s,'blocker','pose_y')])
        d=block-green; d/=np.linalg.norm(d)
        self.green,self.blocker,self.out=green,block,d
        self.theta=self.w(math.atan2(-d[1],-d[0])-self.TOOL_YAW)
        c,z=math.cos(self.theta),math.sin(self.theta)
        self.off=np.array([c*self.OFF[0]-z*self.OFF[1],z*self.OFF[0]+c*self.OFF[1]])
        self.stage=0; self.route=[];self.west_branch=False;self.east_tangent=False
        t=self.target(self.blocker)
        names=s.get_object_names()
        spares=sorted(n for n in names if n.startswith('green') and n!='green0')
        # Some pen orientations put the ideal base pose beyond its +x limit.
        # When available, the unobstructed far-table spare is then preferable.
        if spares and (t[0]>4.60 or d[1]<-.72 or (d[0]<-.65 and d[1]>.60)):
            n=max(spares,key=lambda x:self.g(s,x,'pose_x'))
            self.spare=np.array([self.g(s,n,'pose_x'),self.g(s,n,'pose_y')])
            self.theta=math.pi;self.off=-self.OFF.copy();self.stage=-5
            ft=self.target(self.spare)
            self.far_route=[ft]
            return
        # The exposed blocker can often be taken directly from the west even
        # when following the pen axis would put the base inside the table.
        if not spares and d[0]>.84 and block[0]>=4.61:
            ox=block[0]-5.;length=float(np.linalg.norm(self.OFF))
            oy=-math.sqrt(max(0.,length*length-ox*ox))
            self.theta=self.w(math.atan2(oy,ox)-math.atan2(self.OFF[1],self.OFF[0]))
            c,z=math.cos(self.theta),math.sin(self.theta)
            self.off=np.array([c*self.OFF[0]-z*self.OFF[1],z*self.OFF[0]+c*self.OFF[1]])
            t=self.target(self.blocker);self.east_tangent=True
        elif not spares and d[1]<-.65 and block[0]<4.64:
            self.theta=0.;self.off=self.OFF.copy();t=self.target(self.blocker)
            self.west_branch=True
        elif not spares and t[0]>5.0 and d[1]>.50:
            self.theta=-math.pi/2
            self.off=np.array([self.OFF[1],-self.OFF[0]])
            t=self.target(self.blocker)
        elif not spares and t[0]>5.0 and d[1]<-.30:
            self.theta=.15 if d[1]>-.55 else 0.
            c,z=math.cos(self.theta),math.sin(self.theta)
            self.off=np.array([c*self.OFF[0]-z*self.OFF[1],z*self.OFF[0]+c*self.OFF[1]])
            t=self.target(self.blocker)
            self.west_branch=True
        elif not spares and t[0]>5.0 and block[0]<4.60:
            if d[1]>=0.:
                self.theta=-math.pi/2
                self.off=np.array([self.OFF[1],-self.OFF[0]])
            else:
                self.theta=.15
                c,z=math.cos(self.theta),math.sin(self.theta)
                self.off=np.array([c*self.OFF[0]-z*self.OFF[1],z*self.OFF[0]+c*self.OFF[1]])
                self.west_branch=True
            t=self.target(self.blocker)
        if t[0]>3.65:
            side=1.60 if t[1]>=0 else -1.60
            self.approach_route=[np.array([3.4,side]),np.array([t[0],side]),t]
        else:self.approach_route=[t]
    def robot(self,s): return np.array([self.g(s,'robot','base_x'),self.g(s,'robot','base_y')])
    def target(self,p): return np.asarray(p)-self.off
    def motion(self,s,target,grip=0.,lift=False,arm=True,q4add=0.):
        a=np.zeros(11,dtype=self.dtype); p=self.robot(s)
        a[:2]=np.clip(np.asarray(target)-p,-.2,.2)
        a[2]=np.clip(self.w(self.theta-self.g(s,'robot','base_rot')),-.2,.2)
        if arm:
            for j in range(7):
                goal=self.Q[j]-(.20 if lift and j==1 else 0.)+(q4add if j==3 else 0.)
                d=goal-self.g(s,'robot','joint_'+str(j+1))
                if j in (4,6): d=self.w(d)
                a[3+j]=np.clip(d,-.2,.2)
        a[10]=grip; return a
    def at(self,s,target,arm=False):
        if np.max(np.abs(self.robot(s)-target))>=.004:return False
        if abs(self.w(self.theta-self.g(s,'robot','base_rot')))>=.004:return False
        if arm:
            for j in range(7):
                d=self.Q[j]-self.g(s,'robot','joint_'+str(j+1))
                if j in (4,6):d=self.w(d)
                if abs(d)>=.004:return False
        return True
    def get_action(self,s):
        if self.stage==-5:
            t=self.far_route[0];final=len(self.far_route)==1
            if self.at(s,t,final):
                if final:self.stage=-4
                else:self.far_route.pop(0)
            else:return self.motion(s,t,1,arm=True)
        if self.stage==-4:self.stage=-3;return self.motion(s,self.target(self.spare),-1)
        if self.stage==-3:
            if abs(self.g(s,'robot','joint_2')-(self.Q[1]-.2))<.004:self.stage=-2
            else:return self.motion(s,self.target(self.spare),lift=True)
        if self.stage==-2:
            out=self.spare+np.array([.55,0.]);t=self.target(out)
            if self.at(s,t):
                self.carry_corner=np.array([3.30,-1.55])
                self.route=[np.array([out[0],-1.55]),self.carry_corner]
                self.stage=-1
            else:return self.motion(s,t,lift=True)
        if self.stage==-1:
            if not self.route:self.stage=9
            else:
                t=self.target(self.route[0])
                if self.at(s,t):self.route.pop(0)
                else:return self.motion(s,t,lift=True)
        if self.stage==0:
            t=self.approach_route[0]
            final=len(self.approach_route)==1
            if self.at(s,t,final):
                if len(self.approach_route)>1:self.approach_route.pop(0)
                else:self.stage=1
            else:return self.motion(s,t,1,arm=final)
        if self.stage==1:self.stage=2;return self.motion(s,self.target(self.blocker),-1)
        if self.stage==2:
            # Lift vertically above the wall tops before withdrawing.
            q2=self.g(s,'robot','joint_2')
            if abs(q2-(self.Q[1]-.2))<.004:self.stage=3
            else:return self.motion(s,self.target(self.blocker),lift=True)
        if self.stage==3:
            if self.east_tangent:
                t=self.target(self.blocker)+np.array([0.,.36])
            elif self.out[1]<-.95 and abs(self.out[0])<.16:
                t=self.target(self.blocker)+.70*self.out
            elif self.west_branch and self.out[0]>0. and self.out[1]<0.:
                t=self.target(self.blocker)+np.array([0.,-.62])
            elif self.out[0]<-.60 and self.out[1]>.60:
                t=self.target(self.blocker+.62*self.out)
            else:t=self.target(self.blocker+.36*self.out)
            t[0]=np.clip(t[0],-5.,5.)
            if self.at(s,t):self.stage=4
            else:return self.motion(s,t,lift=True)
        if self.stage==4:self.stage=5;return self.motion(s,self.robot(s),1,lift=True)
        if self.stage==5:
            # A refused release leaves the blocker attached.  Continue carrying
            # it away from the pen before retrying rather than corrupting the
            # following green-block approach.
            if self.g(s,'blocker','grasp_active')>.5:
                if self.east_tangent: direction=np.array([0.,1.])
                elif self.west_branch and self.out[0]>0.: direction=np.array([0.,-1.])
                else: direction=self.out
                return self.motion(s,self.robot(s)+.25*direction,1,lift=True)
            if self.east_tangent:
                if not hasattr(self,'tangent_alt'):
                    ox=self.green[0]-5.;length=float(np.linalg.norm(self.OFF))
                    oy=-math.sqrt(max(0.,length*length-ox*ox))
                    self.theta=self.w(math.atan2(oy,ox)-math.atan2(self.OFF[1],self.OFF[0]))
                    c,z=math.cos(self.theta),math.sin(self.theta)
                    self.off=np.array([c*self.OFF[0]-z*self.OFF[1],z*self.OFF[0]+c*self.OFF[1]])
                    self.tangent_standard=np.r_[self.target(self.green),self.theta]
                    self.tangent_alt,self.tangent_q=self._tangent_ik(self.tangent_standard)
                if self.g(s,'robot','grasp_active')>.5:
                    self.green_grasp_base=self.robot(s).copy();self.stage=7
                elif not hasattr(self,'tangent_shift'):
                    t=self.tangent_standard[:2]
                    if np.max(np.abs(self.robot(s)-t))<.004 and abs(self.w(self.theta-self.g(s,'robot','base_rot')))<.004:
                        self.tangent_shift=True
                    else:return self.motion(s,t,1,lift=True)
                if self.stage==5:
                    t=self.tangent_alt[:2]
                    if np.max(np.abs(self.robot(s)-t))>=.004:
                        return self.motion(s,t,1,lift=True)
                    a=np.zeros(11,dtype=self.dtype)
                    for j in range(7):
                        d=self.tangent_q[j]-self.g(s,'robot','joint_'+str(j+1))
                        if j in (4,6):d=self.w(d)
                        a[3+j]=np.clip(d,-.015,.015)
                    a[10]=-1.;return a
            elif self.west_branch:
                if self.out[0]>=0.:
                    q4=self.g(s,'robot','joint_4')
                    if abs(q4-(self.Q[3]+.08))>=.004:
                        a=np.zeros(11,dtype=self.dtype)
                        a[6]=np.clip(self.Q[3]+.08-q4,-.2,.2);a[10]=1.
                        return a
                    if abs(self.g(s,'robot','joint_2')-self.Q[1])>=.004:
                        return self.motion(s,self.robot(s),1,q4add=.08)
                    self.stage=55
                if not hasattr(self,'west_route'):
                    self.west_route=[self.target(self.green+.31*self.out),
                                     self.target(self.blocker)]
                if self.west_route:
                    t=self.west_route[0]
                    if self.at(s,t):self.west_route.pop(0)
                    else:return self.motion(s,t,1,lift=True)
                else:
                    t=self.target(self.blocker)
                    if self.at(s,t,True):self.stage=55
                    else:return self.motion(s,t,1)
            else:
                t=self.target(self.green)
                if self.at(s,t,True):self.stage=6
                else:return self.motion(s,t,1)
        if self.stage==55:
            q4=self.g(s,'robot','joint_4')
            if self.out[0]<0.:
                t=self.target(self.blocker)+(self.green-self.blocker)+np.array([.006,-.041])
                if self.at(s,t) and abs(q4-(self.Q[3]+.08))<.004:self.stage=56
                else:return self.motion(s,t,1,q4add=.08)
            if abs(q4-(self.Q[3]+.08))>=.004:
                a=np.zeros(11,dtype=self.dtype)
                a[6]=np.clip(self.Q[3]+.08-q4,-.2,.2)
                a[10]=1.;return a
            if not hasattr(self,'se_green_route'):
                c,z=math.cos(self.theta),math.sin(self.theta)
                delta=np.array([c*.10+z*.125,z*.10-c*.125])
                final=self.target(self.green)+delta
                self.se_green_route=[np.array([final[0],self.robot(s)[1]]),final]
            t=self.se_green_route[0]
            if self.at(s,t):
                self.se_green_route.pop(0)
                if not self.se_green_route:self.stage=56
            else:return self.motion(s,t,1,q4add=.08)
        if self.stage==56:
            self.green_grasp_base=self.robot(s).copy()
            self.stage=7
            return self.motion(s,self.robot(s),-1,q4add=.08)
        if self.stage==6:
            self.green_grasp_base=self.robot(s).copy()
            self.stage=7
            return self.motion(s,self.target(self.green),-1)
        if self.stage==7:
            if not hasattr(self,'green_lifted'):
                if abs(self.g(s,'robot','joint_2')-(self.Q[1]-.2))<.004:self.green_lifted=True
                else:
                    t=self.robot(s) if self.west_branch else self.target(self.green)
                    return self.motion(s,t,lift=True,q4add=(.08 if self.west_branch else 0.))
            out=self.green+.62*self.out
            if self.east_tangent:
                t=self.green_grasp_base+np.array([0.,.62])
            elif self.west_branch and self.out[0]>=0.:
                t=self.green_grasp_base+np.array([0.,-.62])
            else:t=self.target(out)
            t[0]=np.clip(t[0],-5.,5.)
            if self.at(s,t):
                out=np.array([self.g(s,'green0','pose_x'),self.g(s,'green0','pose_y')])
                side=.90 if self.east_tangent else (.72 if self.out[1]>=0 else -.72)
                self.route=[np.array([out[0],side]),np.array([2.50,side])]
                if side>0:self.route.append(np.array([2.50,-.72]))
                self.carry_corner=np.array([2.50,-.72])
                self.stage=8
            else:return self.motion(s,t,lift=True,q4add=(.08 if self.west_branch else 0.))
        if self.stage==8:
            if not self.route:self.stage=9
            else:
                t=self.target(self.route[0])
                if self.at(s,t):self.route.pop(0)
                else:return self.motion(s,t,lift=True)
        if self.stage==9:
            self.theta=0.;self.off=self.OFF.copy();t=self.target(self.carry_corner)
            if self.at(s,t):self.stage=10
            else:return self.motion(s,t,lift=True)
        if self.stage==10:
            t=self.target([4.5,-.3])
            if self.at(s,t):self.stage=11
            else:return self.motion(s,t,lift=True)
        if self.stage==11:
            if not hasattr(self,'plate_open_tried'):
                self.plate_open_tried=True
                return self.motion(s,self.target([4.5,-.3]),1,lift=True)
            # Reaching this call means the centered drop was refused.  Use the
            # plate corner farthest from the blocker, preserving ample margins.
            bp=np.array([self.g(s,'blocker','pose_x'),self.g(s,'blocker','pose_y')])
            corners=[np.array([x,y]) for x in (4.30,4.70) for y in (-.50,-.10)]
            goal=max(corners,key=lambda x:float(np.linalg.norm(x-bp)))
            t=self.target(goal)
            if self.at(s,t):return self.motion(s,t,1,lift=True)
            return self.motion(s,t,0,lift=True)
        return np.zeros(11,dtype=self.dtype)
