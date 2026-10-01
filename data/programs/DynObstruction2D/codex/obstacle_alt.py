import math
import numpy as np


class GeneratedApproach:
    """Clear pad blockers outward, then use the baseline cargo sweep."""

    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=float)
        self.high = np.asarray(action_space.high, dtype=float)
        self.robot_type = observation_space.get_type("kin_robot")
        self.block_type = observation_space.get_type("target_block")
        self.surface_type = observation_space.get_type("target_surface")
        self.dyn_type = observation_space.get_type("dyn_rectangle")
        self.reset(None, None)

    def reset(self, state, info):
        self.phase = "clear_select"
        self.phase_step = self.stable = 0
        self.direction = 1
        self.clear_name = None
        self.cleared = set()

    @staticmethod
    def _angle_error(want, have):
        return (want-have+math.pi)%(2*math.pi)-math.pi

    def _advance(self, phase):
        self.phase, self.phase_step, self.stable = phase, 0, 0

    def _objects(self, state, typ):
        return list(state.get_objects(typ))

    def _named(self, state, name):
        try: return state.get_object_from_name(name)
        except Exception: return None

    def get_action(self, state):
        r=self._objects(state,self.robot_type)[0]; b=self._objects(state,self.block_type)[0]; q=self._objects(state,self.surface_type)[0]
        g=lambda o,f: float(state.get(o,f))
        rx,ry,th=g(r,"x"),g(r,"y"),g(r,"theta")
        bx,by,bw,bh=g(b,"x"),g(b,"y"),g(b,"width"),g(b,"height")
        sx,sw=g(q,"x"),g(q,"width")
        a=np.zeros(5,float)

        if self.phase == "clear_select":
            candidates=[]
            lo,hi=sx-sw/2-.015,sx+sw/2+.015
            for o in self._objects(state,self.dyn_type):
                name=getattr(o,"name",str(o))
                if name == "target_block" or name in self.cleared: continue
                ox,ow=g(o,"x"),g(o,"width")
                if ox+ow/2 > lo and ox-ow/2 < hi:
                    candidates.append((abs(ox-sx),name,o))
            if not candidates:
                self.direction=1 if sx>=bx else -1
                self._advance("raise")
            else:
                _,self.clear_name,o=min(candidates)
                ox=g(o,"x")
                # Clear toward its nearest pad edge; tie breaks away from cargo.
                if abs(ox-(sx-sw/2)) < abs((sx+sw/2)-ox): self.direction=-1
                elif abs(ox-(sx-sw/2)) > abs((sx+sw/2)-ox): self.direction=1
                else: self.direction=1 if bx < ox else -1
                self._advance("clear_raise")

        elif self.phase == "clear_raise":
            a[1]=np.clip(1.12-ry,-.049,.049); a[3]=-.099
            a[4]=-.019 if g(r,"finger_gap")>.125 else 0
            if abs(ry-1.12)<.012 or self.phase_step>=30: self._advance("clear_align")

        elif self.phase == "clear_align":
            o=self._named(state,self.clear_name)
            if o is None: self._advance("clear_select")
            else:
                ox,ow=g(o,"x"),g(o,"width")
                # Vertical arm, with narrow closed fingers just beyond the outer face.
                tx=ox-self.direction*(ow/2+.105)
                a[0]=np.clip(tx-rx,-.049,.049)
                a[2]=np.clip(self._angle_error(-math.pi/2,th),-.19,.19); a[3]=-.099
                good=abs(tx-rx)<.012 and abs(self._angle_error(-math.pi/2,th))<.025
                self.stable=self.stable+1 if good else 0
                if self.stable>=2 or self.phase_step>=70: self._advance("clear_lower")

        elif self.phase == "clear_lower":
            o=self._named(state,self.clear_name)
            if o is None: self._advance("clear_select")
            else:
                oy,oh=g(o,"y"),g(o,"height")
                ty=oy+.35-.20*oh
                a[1]=np.clip(ty-ry,-.049,.049); a[3]=-.099
                if abs(ty-ry)<.012 or self.phase_step>=35: self._advance("clear_sweep")

        elif self.phase == "clear_sweep":
            o=self._named(state,self.clear_name)
            if o is None: self._advance("clear_select")
            else:
                ox,ow=g(o,"x"),g(o,"width")
                goal=(sx-self.direction*sw/2)+self.direction*(sw+ow)/2
                # Equivalent: left object entirely left / right object entirely right,
                # with 0.10 clearance. Stop from observation rather than fixed count.
                goal=sx+self.direction*(sw/2+ow/2+.10)
                a[0]=self.direction*.018
                done=(self.direction*(ox-goal)>=0)
                if done or self.phase_step>=75:
                    self.cleared.add(self.clear_name); self._advance("clear_raise")
                    # one raised transition before choosing again
                    self.phase="clear_recover"

        elif self.phase == "clear_recover":
            a[1]=np.clip(1.12-ry,-.049,.049); a[3]=-.099
            if abs(ry-1.12)<.012 or self.phase_step>=30: self._advance("clear_select")

        elif self.phase == "raise":
            a[1],a[3]=np.clip(1.10-ry,-.049,.049),-.099
            if self.direction<0 and bh<.30 and g(r,"finger_gap")<.319: a[4]=.019
            if abs(ry-1.10)<.012 or self.phase_step>=28:self._advance("align")
        elif self.phase == "align":
            if self.direction>0: tx,want=bx-.45,0.0
            elif bh<.30: tx,want=bx+bw/2-.24,-1.20
            else: tx,want=bx+bw/2+.10,-math.pi/2
            a[0]=np.clip(tx-rx,-.049,.049);a[2]=np.clip(self._angle_error(want,th),-.19,.19);a[3]=-.099
            if self.direction<0 and bh>=.30 and g(r,"finger_gap")>.121:a[4]=-.019
            good=abs(tx-rx)<.012 and abs(self._angle_error(want,th))<.025
            self.stable=self.stable+1 if good else 0
            if self.stable>=2 or self.phase_step>=72:self._advance("lower")
        elif self.phase == "lower":
            if self.direction>0:ty=max(.24,by-.20*bh)
            elif bh<.30:ty=by+.215
            else:ty=by+.35-.20*bh
            a[1]=np.clip(ty-ry,-.049,.049);a[3]=-.099
            if abs(ty-ry)<.012 or self.phase_step>=35:self._advance("close" if self.direction>0 else "push")
        elif self.phase == "close":
            if g(r,"finger_gap")>.121:a[4]=-.019
            if self.phase_step>=10:self._advance("push")
        else:
            a[0]=(.010 if self.direction>0 else .005)*self.direction
            if self.phase_step >= (85 if self.direction>0 else 42):self._advance("raise")
        self.phase_step += 1
        return np.clip(a,self.low+1e-7,self.high-1e-7).astype(float)
