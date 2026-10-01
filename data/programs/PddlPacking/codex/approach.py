"""Kinematic pick-and-place policy for PR2Packed."""
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class GeneratedApproach:
    Q_HOME = np.array([.677170, -.343132, 1.2, -1.466884, 1.242232, -1.954428, 2.222541])
    AXES = "zyxyxyx"
    LINKS = (.1, .4, 0., .321, 0., .1, .18)

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        _, self.down_rotation = self._fk(self.Q_HOME)

    @staticmethod
    def _wrap(x):
        return (x + math.pi) % (2 * math.pi) - math.pi

    def _fk(self, q):
        p = np.array([-.05, .188, 1.0])
        r = np.eye(3)
        for axis, angle, length in zip(self.AXES, q, self.LINKS):
            r = r @ Rotation.from_euler(axis, angle).as_matrix()
            p += r @ np.array([length, 0., 0.])
        return p, r

    def _ik(self, xyz, rotation=None, seed=None):
        seed = self.Q_HOME.copy() if seed is None else np.asarray(seed).copy()
        rotation = self.down_rotation if rotation is None else rotation
        def residual(q):
            p, r = self._fk(q)
            orient = Rotation.from_matrix(rotation.T @ r).as_rotvec()
            return np.r_[8*(p-xyz), 8*orient, .002*(q-seed)]
        lo = np.array([-.714, -.523, -.8, -2.32, -20., -2.093, -20.])
        hi = np.array([2.285, 1.396, 3.9, -.001, 20., -.001, 20.])
        q = least_squares(residual, np.clip(seed, lo+1e-4, hi-1e-4),
                          bounds=(lo, hi), max_nfev=350).x
        q[4], q[6] = self._wrap(q[4]), self._wrap(q[6])
        return q

    def _g(self, s, name, feat):
        return s.get(s.get_object_from_name(name), feat)

    def _q(self, s):
        return np.array([self._g(s, "robot", "joint_"+str(i)) for i in range(1, 8)])

    def _yaw(self, s, name):
        return self._wrap(2*math.atan2(self._g(s,name,"pose_qz"), self._g(s,name,"pose_qw")))

    def _slots(self, n):
        if n == 1: return [(0., 0.)]
        if n == 2: return [(-.04, 0.), (.04, 0.)]
        if n == 3: return [(-.08, 0.), (0., 0.), (.08, 0.)]
        cols = 2 if n == 4 else 3
        # Nine 7 cm blocks fit at 9 cm pitch with 1 cm plate margin.  The extra
        # clearance is important because yaw feedback has finite tolerance.
        rows = int(math.ceil(n/cols)); dx = .09
        dy = .09
        return [((c-(cols-1)/2)*dx, (r-(rows-1)/2)*dy)
                for r in range(rows) for c in range(cols)][:n]

    def reset(self, state, info):
        names = sorted(n for n in state.get_object_names() if n.startswith("block"))
        self.todo = sorted(names, key=lambda n: (self._g(state,n,"pose_x") > 0,
                                                  self._g(state,n,"pose_x")))
        self.slots = dict(zip(self.todo, self._slots(len(names))))
        self.index = 0; self.phase = "home"; self.side = -1
        self.route_stage = 0; self.ticks = 0

    def _blank(self): return np.zeros(11, dtype=np.float32)

    def _move_arm(self, s, target):
        d = target-self._q(s); d[4] = self._wrap(d[4]); d[6] = self._wrap(d[6])
        if np.max(np.abs(d)) < .012: return None
        a=self._blank(); a[3:10]=np.clip(d,-.2,.2); return a

    def _move_base(self, s, x=None, y=None, rot=None):
        a=self._blank(); es=[]
        if x is not None:
            d=x-self._g(s,"robot","base_x"); a[0]=np.clip(d,-.2,.2); es.append(abs(d))
        if y is not None:
            d=y-self._g(s,"robot","base_y"); a[1]=np.clip(d,-.2,.2); es.append(abs(d))
        if rot is not None:
            d=self._wrap(rot-self._g(s,"robot","base_rot")); a[2]=np.clip(d,-.2,.2); es.append(abs(d))
        return None if not es or max(es)<.008 else a

    def _navigate(self, s, side, target_y, pick_x):
        # Stop one action-step outside the table.  The fingers can descend here;
        # the final 0.20 m base approach is then collision-free around the block.
        tx=pick_x + (-.2 if side<0 else .2); tr=0. if side<0 else math.pi-.002
        if self._g(s,"robot","base_x")*side <= .65:
            if self.route_stage==0:
                a=self._move_base(s,y=-.98)
                if a is not None:return a
                self.route_stage=1
            if self.route_stage==1:
                a=self._move_base(s,x=tx)
                if a is not None:return a
                self.route_stage=2
            if self.route_stage==2:
                a=self._move_base(s,rot=tr)
                if a is not None:return a
                self.route_stage=3
        else:self.route_stage=3
        a=self._move_base(s,x=tx,y=target_y,rot=tr)
        if a is None:self.side=side;self.route_stage=0
        return a

    def get_action(self, s):
        self.ticks += 1
        if self.index>=len(self.todo): return self._blank()
        name=self.todo[self.index]; held=self._g(s,"robot","grasp_active")>.5
        bx=self._g(s,name,"pose_x"); by=self._g(s,name,"pose_y")
        if self.phase=="home":
            a=self._move_arm(s,self.Q_HOME)
            if a is not None:return a
            self.desired_side=-1 if bx<=0 else 1; self.phase="navigate";self.ticks=0
        if self.phase=="navigate":
            ty=by-.19 if self.desired_side<0 else by+.19
            # Keep roughly the calibrated .755 m reach.  This also leaves the
            # base farther from the table for blocks near its far x edge.
            self.pick_base_x = (min(-.801, bx-.755) if self.desired_side<0
                                else max(.801, bx+.755))
            a=self._navigate(s,self.desired_side,ty,self.pick_base_x)
            if a is not None:return a
            lx=abs(bx-self.pick_base_x)
            self.q_pick=self._ik(np.array([lx,.19,.74]))
            # Match the square's yaw before the close approach; otherwise a
            # rotated corner can hit a finger and reject the entire base step.
            self.q_pick[6]=self._wrap(self.q_pick[6]-self._yaw(s,name))
            self.phase="pick";self.ticks=0
        if self.phase=="alt_home":
            a=self._move_arm(s,self.Q_HOME)
            if a is not None:return a
            a=self._move_base(s,x=self.pick_base_x)
            if a is not None:return a
            self.phase="alt_pick";self.ticks=0
        if self.phase=="alt_pick":
            a=self._move_arm(s,self.q_alt)
            # Contact with the target can reject the last few centimetres of the
            # descent; closing at that point is the empirically successful pose.
            if a is not None and self.ticks < 12:return a
            self.phase="close";a=self._blank();a[10]=-1;return a
        if self.phase=="pick":
            a=self._move_arm(s,self.q_pick)
            if a is not None:
                if self.ticks < 12:return a
                alt_seed=self.Q_HOME.copy();alt_seed[2]-=.8
                self.q_alt=self._ik(np.array([abs(bx-self.pick_base_x),.19,.74]),seed=alt_seed)
                self.q_alt[6]=self._wrap(self.q_alt[6]-self._yaw(s,name))
                self.q_pick=self.q_alt;self.phase="alt_home";self.ticks=0
                return self._blank()
            self.phase="approach";self.ticks=0
        if self.phase=="approach":
            if self.ticks > 5:
                alt_seed=self.Q_HOME.copy();alt_seed[2]-=.8
                self.q_alt=self._ik(np.array([abs(bx-self.pick_base_x),.19,.74]),seed=alt_seed)
                self.q_alt[6]=self._wrap(self.q_alt[6]-self._yaw(s,name))
                self.q_pick=self.q_alt
                self.phase="alt_home";self.ticks=0;return self._blank()
            # If exact centering contacts the palm, stop just short on the next
            # attempt; this is still inside the gripper's approach-axis window.
            a=self._move_base(s,x=self.pick_base_x)
            if a is not None:return a
            self.phase="close";a=self._blank();a[10]=-1;return a
        if self.phase=="close":
            if not held:
                a=self._blank();a[10]=1
                self.q_pick=self._ik(np.array([abs(bx-self.pick_base_x),.19,.735]),seed=self.q_pick)
                self.q_pick[6]=self._wrap(self.q_pick[6]-self._yaw(s,name))
                self.phase="pick";return a
            cur=self._q(s);_,rot=self._fk(cur)
            lx=abs(bx-self.pick_base_x)
            self.q_lift=self._ik(np.array([lx,.19,.90]),rotation=rot,seed=cur)
            self.phase="lift";self.ticks=0
        if self.phase=="lift":
            a=self._move_arm(s,self.q_lift)
            if a is not None:return a
            self.phase="yaw";self.ticks=0
        if self.phase=="yaw":
            yaw=self._yaw(s,name)
            if abs(yaw)>.012:
                a=self._blank();a[9]=np.clip(yaw,-.2,.2);return a
            sx,sy=self.slots[name];lx=sx+.801 if self.side<0 else .801-sx
            cur=self._q(s);_,rot=self._fk(cur)
            self.q_place=self._ik(np.array([lx,.19,.90]),rotation=rot,seed=cur)
            self.phase="arm_place"
        if self.phase=="arm_place":
            a=self._move_arm(s,self.q_place)
            if a is not None:return a
            self.phase="base_place"
        if self.phase=="base_place":
            sx,sy=self.slots[name];ty=sy-.19 if self.side<0 else sy+.19
            a=self._move_base(s,x=-.801 if self.side<0 else .801,y=ty,
                              rot=0. if self.side<0 else math.pi-.002)
            if a is not None:return a
            self.phase="servo";self.ticks=0
        if self.phase=="servo":
            sx,sy=self.slots[name];dx=sx-bx;dy=sy-by
            if max(abs(dx),abs(dy))>.006 and self.ticks<8:
                a=self._blank();a[0]=np.clip(dx,-.04,.04);a[1]=np.clip(dy,-.04,.04);return a
            a=self._blank();a[10]=1;self.phase="open";self.ticks=0;return a
        if self.phase=="open":
            if held:
                a=self._blank();a[4]=-.03;a[10]=1;return a
            self.index+=1;self.phase="home";self.ticks=0;return self._blank()
        return self._blank()
