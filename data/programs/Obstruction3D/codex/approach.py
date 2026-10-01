"""Object-centric geometric manipulation policy for Obstruction3D."""
import math
import numpy as np


class GeneratedApproach:
    R = .656
    YAW_GAIN = 1.0
    Q2, CARRY_Q2, Q4, Q6, Q7 = .65, .25, -1.50, -.87, math.pi/2

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space

    @staticmethod
    def _obj(s,n): return s.get_object_from_name(n)
    @classmethod
    def _g(cls,s,n,f): return float(s.get(cls._obj(s,n),f))
    @staticmethod
    def _d(goal,now): return float(np.clip(goal-now,-.2,.2))

    def reset(self,state,info):
        tx,ty=self._g(state,"target_region","pose_x"),self._g(state,"target_region","pose_y")
        thx,thy=self._g(state,"target_region","half_extent_x"),self._g(state,"target_region","half_extent_y")
        obs=[]
        for n in state.get_object_names():
            if n.startswith("obstruction") and abs(self._g(state,n,"pose_x")-tx)<thx+self._g(state,n,"half_extent_x") and abs(self._g(state,n,"pose_y")-ty)<thy+self._g(state,n,"half_extent_y"):
                obs.append(n)
        # Higher pieces go first so stacked/adjacent blocks do not mask grasps.
        order=sorted(obs,key=lambda n:-self._g(state,n,"pose_z"))
        self.todo=order+["target_block"]
        self.i=0; self.stage="plan"; self.wait=0; self.q4_try=0; self.search_pos=0;self.target_sweep=0
        self.recovery_grasp=False

    def _held(self,s):
        for n in s.get_object_names():
            if (n=="target_block" or n.startswith("obstruction")) and self._g(s,n,"grasp_active")>.5:
                return n
        return None

    def _act(self,s,bx=None,by=None,q1=None,q2=None,q4=None,q6=None,q7=None,grip=0.):
        a=np.zeros(11,np.float32)
        for idx,feat,goal in ((0,"pos_base_x",bx),(1,"pos_base_y",by),(3,"joint_1",q1),
                              (4,"joint_2",q2),(6,"joint_4",q4),(8,"joint_6",q6),(9,"joint_7",q7)):
            if goal is not None:a[idx]=self._d(goal,self._g(s,"robot",feat))
        a[10]=grip; return a

    def _at(self,s,f,x,t=.005): return abs(self._g(s,"robot",f)-x)<t

    def _destination(self,s,name):
        if name=="target_block":
            ox,oy=self._g(s,name,"pose_x"),self._g(s,name,"pose_y")
            gx,gy=self._g(s,"target_region","pose_x"),self._g(s,"target_region","pose_y")
            dx,dy=gx-ox,gy-oy
            if abs(dx)<.004 and abs(dy)<.004:
                return gx,gy
            # Keep grasp yaw near zero. When x dominates, make progress through
            # shallow diagonal zigzags; both circle centers then remain in
            # front of the table and the downward gripper stays graspable.
            if abs(dx)>.10*abs(dy):
                sx=float(np.clip(dx,-.012,.012))
                direction=-1. if self.target_sweep%2==0 else 1.
                if oy+direction*.12<-.35 or oy+direction*.12>.35:direction=-direction
                self.target_sweep+=1
                return ox+sx,oy+direction*.12
            scale=min(1.,.10/max(abs(dy),1e-8))
            return ox+dx*scale,oy+dy*scale
        # A short sideways transfer is much less collision-prone than carrying
        # to an edge. Move away from the region and stay inside table bounds.
        ox,oy=self._g(s,name,"pose_x"),self._g(s,name,"pose_y")
        step=.12
        ny=oy-step
        if ny<-.34:ny=oy+step
        return ox,ny

    def _make_plan(self,s,name):
        src=(self._g(s,name,"pose_x"),self._g(s,name,"pose_y")); dst=self._destination(s,name)
        dx,dy=dst[0]-src[0],dst[1]-src[1]; dist=max(math.hypot(dx,dy),1e-8)
        # Intersection of radius-R circles about source and destination.
        # The smaller-x solution keeps the mobile base in front of the table.
        mx,my=(src[0]+dst[0])/2,(src[1]+dst[1])/2
        h=math.sqrt(max(self.R*self.R-dist*dist/4,0.))
        centers=((mx-dy/dist*h,my+dx/dist*h),(mx+dy/dist*h,my-dx/dist*h))
        viable=[p for p in centers if p[0]<.08] or list(centers)
        self.base=min(viable,key=lambda p:abs(math.atan2(dst[1]-p[1],dst[0]-p[0]))); self.dest=dst
        a0=math.atan2(src[1]-self.base[1],src[0]-self.base[0])
        a1=math.atan2(dst[1]-self.base[1],dst[0]-self.base[0])
        self.q_start=-a0
        self.q_goal=self.q_start-(a1-a0)/self.YAW_GAIN
        self.q4_try=0; self.search_pos=0; self.wait=0;self.final_corrected=False

    def get_action(self,s):
        if self.i>=len(self.todo): return self._act(s,grip=1.)
        name=self.todo[self.i]; held=self._held(s)
        if self.stage=="plan":
            self._make_plan(s,name); self.stage="reposition"

        if self.stage=="reposition":
            ready=(self._at(s,"pos_base_x",self.base[0]) and self._at(s,"pos_base_y",self.base[1])
                   and self._at(s,"joint_1",self.q_start) and self._at(s,"joint_2",self.CARRY_Q2)
                   and self._at(s,"joint_4",self.Q4) and self._at(s,"joint_6",self.Q6))
            if ready:self.stage="approach";self.wait=0
            return self._act(s,*self.base,self.q_start,self.CARRY_Q2,self.Q4,self.Q6,self.Q7,grip=1.)

        if self.stage in ("approach","search") and held is not None and held!=name:
            return self._act(s,grip=1.)

        if held==name and self.stage in ("approach","search"):
            self.recovery_grasp=False
            self.grasp_q4=self._g(s,"robot","joint_4")
            self.obstacle_last_q4=None
            self.base=(self._g(s,"robot","pos_base_x"),self._g(s,"robot","pos_base_y"))
            cur=math.atan2(self._g(s,held,"pose_y")-self.base[1],self._g(s,held,"pose_x")-self.base[0])
            want=math.atan2(self.dest[1]-self.base[1],self.dest[0]-self.base[0])
            self.q_goal=self._g(s,"robot","joint_1")-(want-cur)
            self.stage="lift"; self.wait=0

        if self.stage in ("approach","search"):
            # Height scan covers all specified obstruction half-extents.
            heights=(-1.50,-1.42,-1.58,-1.34,-1.66)
            raised_target=(name=="target_block" and
                           (self._g(s,name,"pose_z")>.108 or self.recovery_grasp))
            q4=-1.55 if raised_target else heights[min(self.q4_try,len(heights)-1)]
            grasp_q2=(.62 if self.recovery_grasp else (.60 if raised_target else self.Q2))
            offsets=((0.,0.),(-.02,0.),(.02,0.),(0.,-.02),(0.,.02),(-.03,-.02),(-.03,.02))
            off=offsets[self.search_pos%len(offsets)]; bx=self.base[0]+off[0];by=self.base[1]+off[1]
            ready=(self._at(s,"pos_base_x",bx) and self._at(s,"pos_base_y",by)
                   and self._at(s,"joint_1",self.q_start) and self._at(s,"joint_2",grasp_q2)
                   and self._at(s,"joint_4",q4) and self._at(s,"joint_6",self.Q6))
            if ready:
                self.wait+=1
                if self.wait>3:
                    self.q4_try+=1;self.wait=0;self.stage="search"
                    if self.q4_try>=len(heights):self.q4_try=0;self.search_pos+=1
            commanded_q2=grasp_q2
            target_z=self._g(s,name,"pose_z") if name=="target_block" else 0.
            if name=="target_block" and .105 < target_z <= .108:
                # Large blocks can make a full 0.2-rad descent collision-invalid;
                # close continuously while descending through the grasp band.
                commanded_q2=min(grasp_q2,self._g(s,"robot","joint_2")+.02)
            return self._act(s,bx,by,self.q_start,commanded_q2,q4,self.Q6,self.Q7,grip=-1.)

        if self.stage=="lift":
            if self._at(s,"joint_2",self.CARRY_Q2):
                self.wait+=1
                if self.wait>=2:self.stage="swing";self.wait=0
            return self._act(s,q2=self.CARRY_Q2,q6=self.Q6,q7=self.Q7)

        if self.stage=="swing":
            if self._at(s,"joint_1",self.q_goal):
                final=(name=="target_block" and
                       abs(self.dest[0]-self._g(s,"target_region","pose_x"))<.003 and
                       abs(self.dest[1]-self._g(s,"target_region","pose_y"))<.003)
                if final and not self.final_corrected:
                    ox,oy=self._g(s,name,"pose_x"),self._g(s,name,"pose_y")
                    if math.hypot(ox-self.dest[0],oy-self.dest[1])>.025:
                        cur=math.atan2(oy-self.base[1],ox-self.base[0])
                        want=math.atan2(self.dest[1]-self.base[1],self.dest[0]-self.base[0])
                        self.q_goal=self._g(s,"robot","joint_1")-(want-cur)/self.YAW_GAIN
                        self.final_corrected=True;self.wait=0
                        return self._act(s,q1=self.q_goal,q2=self.CARRY_Q2,q6=self.Q6,
                                         q7=self.Q7+(self.q_goal-self.q_start))
                self.wait+=1
                if self.wait>=2:self.stage="lower";self.wait=0;self.last_lower_q2=None;self.last_lower_q4=None
            q7=self.Q7+(self.q_goal-self.q_start) if name=="target_block" else self.Q7
            return self._act(s,q1=self.q_goal,q2=self.CARRY_Q2,q6=self.Q6,q7=q7)

        if self.stage=="lower":
            if name=="target_block":
                q2=self._g(s,"robot","joint_2");q4=self._g(s,"robot","joint_4")
                q7=self.Q7+(self.q_goal-self.q_start)
                if q2<.595:
                    return self._act(s,q1=self.q_goal,q2=min(.60,q2+.05),q4=self.Q4,q6=self.Q6,q7=q7)
                if q4> -1.545 and self.last_lower_q4 is None:
                    self.last_lower_q4=q4
                    return self._act(s,q1=self.q_goal,q2=.60,q4=-1.55,q6=self.Q6,q7=q7)
                # Descend in 0.01-rad increments. A target pad stops this motion
                # earlier than the table; either contact or q2=.62 is followed
                # immediately by opening.
                final=(abs(self.dest[0]-self._g(s,"target_region","pose_x"))<.003 and
                       abs(self.dest[1]-self._g(s,"target_region","pose_y"))<.003)
                support_z=.11 if final else .10
                object_z=self._g(s,name,"pose_z")
                if object_z<=support_z+.003 or (self.last_lower_q2 is not None and abs(q2-self.last_lower_q2)<.002):
                    self.stage="release"
                    return self._act(s,grip=1.)
                self.last_lower_q2=q2
                return self._act(s,q1=self.q_goal,q2=min(.65,q2+.01),q4=-1.55,q6=self.Q6,q7=q7)
            final_target=(name=="target_block" and
                          abs(self.dest[0]-self._g(s,"target_region","pose_x"))<.002 and
                          abs(self.dest[1]-self._g(s,"target_region","pose_y"))<.002)
            release_q2=(.63 if final_target else self.Q2)
            if self._at(s,"joint_2",release_q2):
                # Opening must be issued immediately after the lowering step.
                # A neutral contact step makes the strict simulator reject all
                # later open commands.
                ox,oy=self._g(s,name,"pose_x"),self._g(s,name,"pose_y")
                tol=.006 if name=="target_block" else .04
                if math.hypot(ox-self.dest[0],oy-self.dest[1])<=tol:
                    desired_z=.075+self._g(s,name,"half_extent_z")
                    cur_q4=self._g(s,"robot","joint_4")
                    high=self._g(s,name,"pose_z")>desired_z+.003
                    stalled=self.obstacle_last_q4 is not None and abs(cur_q4-self.obstacle_last_q4)<.002
                    if high and not stalled:
                        self.obstacle_last_q4=cur_q4
                        return self._act(s,q1=self.q_goal,q2=self.Q2,q4=cur_q4-.01,q6=self.Q6,q7=self.Q7)
                    self.stage="release"
                    return self._act(s,grip=1.)
                cur=math.atan2(oy-self.base[1],ox-self.base[0])
                want=math.atan2(self.dest[1]-self.base[1],self.dest[0]-self.base[0])
                self.q_goal=self._g(s,"robot","joint_1")-(want-cur)/self.YAW_GAIN
                self.stage="lift";self.wait=0
                return self._act(s,q2=self.CARRY_Q2,q6=self.Q6,q7=self.Q7)
            lower_goal=release_q2
            if name=="target_block":
                lower_goal=min(release_q2,self._g(s,"robot","joint_2")+.05)
            return self._act(s,q1=self.q_goal,q2=lower_goal,q6=self.Q6,q7=self.Q7)

        if self.stage=="check":
            ox,oy=self._g(s,name,"pose_x"),self._g(s,name,"pose_y")
            tol=.006 if name=="target_block" else .04
            if math.hypot(ox-self.dest[0],oy-self.dest[1])<=tol:
                self.stage="release"
                return self._act(s,grip=1.)
            # Correct tangential yaw error, then repeat the slight lift/swing.
            cur=math.atan2(oy-self.base[1],ox-self.base[0])
            want=math.atan2(self.dest[1]-self.base[1],self.dest[0]-self.base[0])
            self.q_goal=self._g(s,"robot","joint_1")-(want-cur)/self.YAW_GAIN
            self.stage="lift";self.wait=0
            q7=self.Q7+(self.q_goal-self.q_start) if name=="target_block" else self.Q7
            return self._act(s,q2=self.CARRY_Q2,q6=self.Q6,q7=q7)

        if self.stage=="release":
            if held is None:
                if name=="target_block":
                    gx,gy=self._g(s,"target_region","pose_x"),self._g(s,"target_region","pose_y")
                    ox,oy=self._g(s,name,"pose_x"),self._g(s,name,"pose_y")
                    if math.hypot(gx-ox,gy-oy)>.004:
                        if self.recovery_grasp:
                            self.base=(self._g(s,"robot","pos_base_x"),
                                       self._g(s,"robot","pos_base_y"))
                            self.q_start=self._g(s,"robot","joint_1")
                            self.dest=self._destination(s,name)
                            self.q4_try=0;self.search_pos=0;self.wait=0
                            self.stage="approach"
                            return self._act(s,grip=1.)
                        self.stage="plan";self.wait=0
                        return self._act(s,grip=1.)
                self.i+=1;self.stage="plan";self.wait=0
                return self._act(s,grip=1.)
            # If the first open was rejected, the object is usually a few
            # millimetres above support after a collision-limited descent.
            # Continue tiny wrist descents while opening; the first valid
            # contact step then releases immediately instead of looping.
            cur_q4=self._g(s,"robot","joint_4")
            if held==name and cur_q4> -1.70:
                self.recovery_grasp=True
                return self._act(s,q4=cur_q4-.01,grip=1.)
            return self._act(s,grip=1.)
        return self._act(s)
