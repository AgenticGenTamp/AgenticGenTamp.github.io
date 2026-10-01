"""Adaptive variable-count policy for Transport3DEnv."""
import math
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.reset(None, None)

    def reset(self, state, info):
        self.tick = 0; self.phase = "search"; self.phase_ticks = 0
        self.target = None; self.learned_q = None; self.offset = None; self.drop = 0
        self.route_index = 0
        self.delivered = set()
        self.tool_cube = None
        self.tool_repeat = False
        self.tool_attempt = 0

    def g(self, s, n, f):
        return float(s.get(s.get_object_from_name(n), f))

    def q(self, s):
        return np.array([self.g(s,"robot","joint_%d"%i) for i in range(1,8)])

    def objects(self, s):
        ns=s.get_object_names()
        cs=sorted([n for n in ns if n.startswith("cube")],
                  key=lambda n:int(n[4:]) if n[4:].isdigit() else n)
        return (["box0"] if "box0" in ns else [])+cs

    def on_table(self,s,n):
        x,y,z=[self.g(s,n,"pose_"+c) for c in "xyz"]
        hz=self.g(s,n,"half_extent_z")
        # A kinematic object does not fall after release: it must be at the
        # tabletop height, not merely somewhere above the table footprint.
        return abs((z-hz)-.4)<.035 and .36<=x<=.84 and -.44<=y<=.44

    def choose(self,s):
        return next((n for n in self.objects(s)
                     if n not in self.delivered and not self.on_table(s,n)),None)

    def base(self,s,x,y,a):
        dx=x-self.g(s,"robot","pos_base_x");dy=y-self.g(s,"robot","pos_base_y")
        a[:2]=np.clip([dx,dy],-.2,.2)
        return max(abs(dx),abs(dy))<.025

    def joints(self,s,goal,a):
        d=goal-self.q(s);a[3:10]=np.clip(d,-.2,.2)
        return np.max(np.abs(d))<.035

    def drop_point(self):
        if self.target=="box0": return .6,0.
        return [(0.72,-.28),(0.72,-.10),(0.72,.10),(0.72,.28)][self.drop%4]

    def get_action(self,s):
        self.tick+=1;self.phase_ticks+=1;a=np.zeros(11,np.float32)
        held=self.g(s,"robot","grasp_active")>.5
        target_held=(self.target is not None and
                     self.g(s,self.target,"grasp_active")>.5)
        if target_held and self.phase in ("search","close"):
            self.learned_q=self.q(s).copy()
            self.offset=np.array([self.g(s,self.target,"pose_x")-self.g(s,"robot","pos_base_x"),
                                  self.g(s,self.target,"pose_y")-self.g(s,"robot","pos_base_y")])
            remaining=[n for n in self.objects(s) if n.startswith("cube") and n not in self.delivered]
            if self.target=="box0" and remaining:
                self.tool_cube=remaining[0];self.phase="tool_stage"
            else:self.phase="lift"
            self.phase_ticks=0
        if self.target is None or (not held and self.on_table(s,self.target)):
            if self.target is not None:self.drop+=1
            self.target=self.choose(s)
            if self.target is None:return a
            self.phase="search" if self.learned_q is None else "approach";self.phase_ticks=0
        if self.phase=="search":
            if self.target == "box0":
                # Collision-aware route discovered empirically.  The seemingly
                # redundant waypoints select a reachable winding of the arm;
                # commanding only the final pose is blocked by self-collision.
                qs=[
                 [.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593],
                 [4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916],
                 [4.170779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434]]
                offs=[[-.799987478,-.347800459,2.104792961],
                      [.126212298,-.083534270,-3.139703362],
                      [-.428185141,.054943428,2.978131848]]
                if self.route_index<3:i=self.route_index
                else:i=[1,2,0][(self.route_index-3)%3]
                tx=self.g(s,self.target,"pose_x");ty=self.g(s,self.target,"pose_y")
                self.base(s,tx+offs[i][0],ty+offs[i][1],a)
                a[2]=np.clip(offs[i][2]-self.g(s,"robot","pos_base_rot"),-.2,.2)
                self.joints(s,np.asarray(qs[i]),a)
                a[10]=1 if self.phase_ticks<=35 else -1
                if self.phase_ticks>=37:
                    self.route_index+=1;self.phase_ticks=0
                return a
            # Smooth incommensurate sweeps cover workspace between isolated poses.
            k=self.phase_ticks;tx=self.g(s,self.target,"pose_x");ty=self.g(s,self.target,"pose_y")
            r=.5+.4*math.sin(k*.017)
            self.base(s,tx+r*math.sin(k*.061),ty+r*math.sin(k*.047+1.1),a)
            yaw=math.pi*math.sin(k*.013+.3)
            a[2]=np.clip(yaw-self.g(s,"robot","pos_base_rot"),-.2,.2)
            lo=np.array([0.,-.35,-math.pi,-2.5,0.,-.87,math.pi/2])
            hi=np.array([5.2,2.41,2.05,2.66,5.2,2.23,6.77])
            freq=np.array([.031,.023,.019,.029,.037,.041,.043])
            phase=np.array([0.,.7,1.7,2.1,.4,2.8,1.2])
            goal=lo+(hi-lo)*(.5+.5*np.sin(k*freq+phase))
            self.joints(s,goal,a);a[10]=1 if k%6<2 else -1;return a
        if self.phase=="approach":
            tx=self.g(s,self.target,"pose_x");ty=self.g(s,self.target,"pose_y")
            rb=self.base(s,tx-self.offset[0],ty-self.offset[1],a)
            rq=self.joints(s,self.learned_q,a);a[10]=1
            if rb and rq:self.phase="close";self.phase_ticks=0
            return a
        if self.phase=="close":
            a[10]=-1
            if self.phase_ticks>4:self.phase="approach";self.phase_ticks=0
            return a
        if self.phase=="tool_stage":
            c=np.array([self.g(s,self.tool_cube,"pose_x"),self.g(s,self.tool_cube,"pose_y")])
            u=np.array([.6-c[0],-c[1]]);u/=max(np.linalg.norm(u),1e-6)
            many=len([n for n in self.objects(s) if n.startswith("cube")])>1
            gap=.18 if many else (.14 if self.tool_attempt else .10)
            lateral=np.array([-u[1],u[0]])*(.04 if self.tool_attempt>=2 else 0.)
            side=c-u*gap+lateral;p=np.array([self.g(s,"box0","pose_x"),self.g(s,"box0","pose_y")])
            bx=self.g(s,"robot","pos_base_x");by=self.g(s,"robot","pos_base_y")
            self.base(s,bx+side[0]-p[0],by+side[1]-p[1],a)
            if not self.tool_repeat:self.joints(s,self.learned_q,a)
            a[10]=-1
            if self.phase_ticks>=15:self.phase="tool_sweep";self.phase_ticks=0
            return a
        if self.phase=="tool_sweep":
            p=np.array([self.g(s,"box0","pose_x"),self.g(s,"box0","pose_y")])
            a[:2]=np.clip(np.array([.6,0.])-p,-.08,.08)
            a[4]=-.04 if self.tool_attempt else -.10;a[10]=-1
            if self.phase_ticks>=10:self.phase="tool_carry";self.phase_ticks=0
            return a
        if self.phase=="tool_carry":
            c=np.array([self.g(s,self.tool_cube,"pose_x"),self.g(s,self.tool_cube,"pose_y")])
            a[:2]=np.clip(np.array([.6,0.])-c,-.10,.10);a[10]=-1
            if self.phase_ticks>=35:self.phase="tool_settle2";self.phase_ticks=0
            return a
        if self.phase=="tool_settle2":
            c=np.array([self.g(s,self.tool_cube,"pose_x"),self.g(s,self.tool_cube,"pose_y")])
            a[:2]=np.clip(np.array([.6,0.])-c,-.08,.08);a[4]=.10;a[10]=-1
            if self.phase_ticks>=12:self.phase="tool_settle4";self.phase_ticks=0
            return a
        if self.phase=="tool_settle4":
            c=np.array([self.g(s,self.tool_cube,"pose_x"),self.g(s,self.tool_cube,"pose_y")])
            a[:2]=np.clip(np.array([.6,0.])-c,-.08,.08);a[6]=-.10;a[10]=-1
            if self.phase_ticks>=8:
                many=len([n for n in self.objects(s) if n.startswith("cube")])>1
                if not many and not self.on_table(s,self.tool_cube) and self.tool_attempt<2:
                    # The cube can remain supported high on the carried box.
                    # Releasing, then stepping the mobile base diagonally,
                    # empirically clears that support and lets both bodies
                    # settle onto the tabletop.
                    self.phase="tool_seat_release";self.phase_ticks=0
                    return a
                self.delivered.add(self.tool_cube)
                more=[n for n in self.objects(s) if n.startswith("cube") and n not in self.delivered]
                if more:
                    self.tool_cube=more[0];self.tool_repeat=True;self.phase="tool_stage"
                elif len([n for n in self.objects(s) if n.startswith("cube")])>1:
                    self.phase="tool_boxfinish"
                else:self.phase="release"
                self.phase_ticks=0
            return a
        if self.phase=="tool_seat_release":
            a[10]=1
            if self.phase_ticks>=2:
                a[0]=.2;a[1]=.2
            return a
        if self.phase=="tool_boxfinish":
            p=np.array([self.g(s,"box0","pose_x"),self.g(s,"box0","pose_y")])
            a[:2]=np.clip(np.array([.74,0.])-p,-.10,.10);a[10]=-1
            if self.phase_ticks>=20:self.phase="release";self.phase_ticks=0
            return a
        if self.phase=="lift":
            q=self.learned_q.copy();q[1]=max(-.35,q[1]-1.6);self.joints(s,q,a);a[10]=-1
            if self.phase_ticks>=10:self.phase="carry";self.phase_ticks=0
            return a
        if self.phase=="carry":
            x,y=self.drop_point();ox=self.g(s,self.target,"pose_x");oy=self.g(s,self.target,"pose_y")
            bx=self.g(s,"robot","pos_base_x");by=self.g(s,"robot","pos_base_y")
            arrived=self.base(s,bx+x-ox,by+y-oy,a)
            q=self.learned_q.copy();q[1]=max(-.35,q[1]-1.6);self.joints(s,q,a);a[10]=-1
            if arrived:self.phase="lower";self.phase_ticks=0
            return a
        if self.phase=="lower":
            x,y=self.drop_point();ox=self.g(s,self.target,"pose_x");oy=self.g(s,self.target,"pose_y")
            bx=self.g(s,"robot","pos_base_x");by=self.g(s,"robot","pos_base_y")
            self.base(s,bx+x-ox,by+y-oy,a)
            done=self.joints(s,self.learned_q,a);a[10]=-1
            if self.phase_ticks>=12:self.phase="release";self.phase_ticks=0
            return a
        if self.phase=="release":
            a[10]=1
            if not held and self.phase_ticks>2:
                self.delivered.add(self.target)
                # Keep optimistic bookkeeping during the coupled box/cube
                # maneuver, but reconcile it after release if the episode did
                # not terminate.  This prevents a failed placement from
                # leaving the controller idle for the rest of the horizon.
                for name in self.objects(s):
                    if name.startswith("cube") and not self.on_table(s,name):
                        self.delivered.discard(name)
                # The large-box contact pose is vertically wrong for 5 cm
                # cubes; force fresh workspace exploration for the next type.
                if self.target == "box0":
                    self.learned_q=None;self.offset=None
                self.target=None
            return a
        return a
