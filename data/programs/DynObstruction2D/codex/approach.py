import numpy as np
import math


class GeneratedApproach:
    """Repeatedly stages a finger behind the block and sweeps it onto the pad."""

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
        self.phase_step = 0
        self.direction = 1
        self.stable = 0
        self.clear_name = None
        self.cleared = set()
        self.boundary_pick = False
        if state is not None:
            r = state.get_objects(self.robot_type)[0]
            b = state.get_objects(self.block_type)[0]
            q = state.get_objects(self.surface_type)[0]
            g = lambda o, f: float(state.get(o, f))
            has_obstructions = any(name.startswith("obstruction")
                                   for name in state.get_object_names())
            pad_blocked = False
            for obj in state.get_objects(self.dyn_type):
                if self._name(obj) != "target_block":
                    depth = self._overlap_depth(g(obj, "x"), g(obj, "width"),
                                                g(q, "x"), g(q, "width"))
                    pad_blocked = pad_blocked or depth > 0.0
            # A normal left push cannot stage behind narrow cargo against the
            # right wall.  When it fits between fully open fingers, pick it up
            # from above and place it directly instead.
            fits = g(b, "width") < g(r, "finger_gap") - .005
            boundary_left = g(q, "x") < g(b, "x") and g(b, "x") > 2.70
            distant_narrow = (g(b, "width") < .22 and
                              abs(g(q, "x") - g(b, "x")) > .60)
            self.boundary_pick = (not pad_blocked and fits and
                                  (boundary_left or distant_narrow))
            if self.boundary_pick:
                self.phase = "boundary_raise"

    @staticmethod
    def _angle_error(want, have):
        return (want - have + math.pi) % (2.0 * math.pi) - math.pi

    def _advance(self, phase):
        self.phase = phase
        self.phase_step = 0
        self.stable = 0

    @staticmethod
    def _name(obj):
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _overlap_depth(x1, w1, x2, w2):
        return min(x1 + w1 / 2.0, x2 + w2 / 2.0) - max(
            x1 - w1 / 2.0, x2 - w2 / 2.0)

    def _named(self, state, name):
        try:
            return state.get_object_from_name(name)
        except Exception:
            return None

    def get_action(self, state):
        r = state.get_objects(self.robot_type)[0]
        b = state.get_objects(self.block_type)[0]
        q = state.get_objects(self.surface_type)[0]
        g = lambda o, f: float(state.get(o, f))
        rx, ry, th = g(r, "x"), g(r, "y"), g(r, "theta")
        bx, by, bw = g(b, "x"), g(b, "y"), g(b, "width")
        bh = g(b, "height")
        sx, sw = g(q, "x"), g(q, "width")
        wide_low = bh < .30 and bw > .35
        has_obstructions = any(name.startswith("obstruction")
                               for name in state.get_object_names())
        pad_blocked_now = any(
            self._name(obj) != "target_block" and g(obj, "y") < .8 and
            self._overlap_depth(g(obj, "x"), g(obj, "width"), sx, sw) > .02
            for obj in state.get_objects(self.dyn_type))
        short_clear_carry = abs(bx - sx) < 1.0 and not pad_blocked_now
        narrow_left = (self.direction < 0 and
                       ((not has_obstructions and bw < .26) or
                        (has_obstructions and bw < .22 and short_clear_carry)))
        horizontal_left = (self.direction < 0 and bx < 2.70 and bh > .44
                           and not has_obstructions)
        a = np.zeros(5, dtype=float)

        if self.phase == "boundary_raise":
            a[1] = np.clip(1.12 - ry, -.049, .049)
            a[3], a[4] = -.099, .019
            if abs(ry - 1.12) < .012 or self.phase_step >= 30:
                self._advance("boundary_align")

        elif self.phase == "boundary_align":
            error = self._angle_error(-math.pi / 2.0, th)
            a[0] = np.clip(bx - rx, -.049, .049)
            a[2] = np.clip(error, -.19, .19)
            a[3], a[4] = -.099, .019
            if ((abs(bx - rx) < .01 and abs(error) < .02)
                    or self.phase_step >= 70):
                self._advance("boundary_lower")

        elif self.phase == "boundary_lower":
            target_y = by + .40
            a[1] = np.clip(target_y - ry, -.049, .049)
            a[3], a[4] = -.099, .019
            if abs(target_y - ry) < .01 or self.phase_step >= 35:
                self._advance("boundary_close")

        elif self.phase == "boundary_close":
            a[4] = -.019
            if g(b, "held") > .5:
                self._advance("boundary_lift")
            elif self.phase_step >= 14:
                # Never execute a lift/transport unless attachment succeeded.
                self.boundary_pick = False
                self._advance("clear_select")

        elif self.phase == "boundary_lift":
            a[1] = np.clip(.62 - by, -.035, .035)
            if abs(by - .62) < .012 or self.phase_step >= 30:
                self._advance("boundary_translate")

        elif self.phase == "boundary_translate":
            a[0] = np.clip(sx - bx, -.025, .025)
            if abs(sx - bx) < .008 or self.phase_step >= 120:
                self._advance("boundary_place")

        elif self.phase == "boundary_place":
            target_y = (g(q, "y") + g(q, "height") / 2.0
                        + bh / 2.0 + .006)
            a[1] = np.clip(target_y - by, -.025, .025)
            if abs(target_y - by) < .008 or self.phase_step >= 30:
                self._advance("boundary_release")

        elif self.phase == "boundary_release":
            a[4] = .019
            if g(b, "held") < .5 or self.phase_step >= 14:
                self._advance("boundary_withdraw")

        elif self.phase == "boundary_withdraw":
            a[1] = .035

        elif self.phase == "clear_select":
            obstructions = [o for o in state.get_objects(self.dyn_type)
                            if self._name(o) != "target_block"]
            target_overlaps = self._overlap_depth(bx, bw, sx, sw) > 0.0
            single_right_blocked = (len(obstructions) == 1 and bx < sx and
                                    bw > .20 and bh > .30 and
                                    g(obstructions[0], "width") > bw)
            single_left_edge = (len(obstructions) == 1 and bx > sx and sx < .60
                                and g(obstructions[0], "x") < sx)
            candidates = []
            if not wide_low and (len(obstructions) >= 2 or target_overlaps or
                                 single_right_blocked or single_left_edge):
                for o in obstructions:
                    name = self._name(o)
                    if name in self.cleared:
                        continue
                    ox, ow = g(o, "x"), g(o, "width")
                    if self._overlap_depth(ox, ow, sx, sw) > .04:
                        candidates.append((abs(ox - sx), name, o))
            # On a very wide pad, direct placement has ample room beside a
            # blocker. Multiple low blockers are likewise safer to bulldoze
            # with the cargo than to rake (their top faces catch the palm).
            if sw > .55 or (len(candidates) >= 2 and
                            min(g(item[2], "height") for item in candidates) < .30):
                candidates = []
            if not candidates:
                self.direction = 1 if sx >= bx else -1
                self._advance("raise")
            else:
                _, self.clear_name, o = min(candidates)
                ox = g(o, "x")
                left_distance = abs(ox - (sx - sw / 2.0))
                right_distance = abs((sx + sw / 2.0) - ox)
                if left_distance < right_distance:
                    self.direction = -1
                elif right_distance < left_distance:
                    self.direction = 1
                else:
                    self.direction = 1 if bx < ox else -1
                self._advance("clear_raise")

        elif self.phase == "clear_raise":
            a[1], a[3] = np.clip(1.12 - ry, -.049, .049), -.099
            if g(r, "finger_gap") > .125:
                a[4] = -.019
            if abs(ry - 1.12) < .012 or self.phase_step >= 30:
                self._advance("clear_align")

        elif self.phase == "clear_align":
            o = self._named(state, self.clear_name)
            if o is None:
                self._advance("clear_select")
            else:
                ox, ow = g(o, "x"), g(o, "width")
                tx = ox - self.direction * (ow / 2.0 + .105)
                a[0] = np.clip(tx - rx, -.049, .049)
                a[2] = np.clip(self._angle_error(-math.pi / 2.0, th), -.19, .19)
                a[3] = -.099
                good = (abs(tx - rx) < .012 and
                        abs(self._angle_error(-math.pi / 2.0, th)) < .025)
                self.stable = self.stable + 1 if good else 0
                if self.stable >= 2 or self.phase_step >= 70:
                    self._advance("clear_lower")

        elif self.phase == "clear_lower":
            o = self._named(state, self.clear_name)
            if o is None:
                self._advance("clear_select")
            else:
                ty = g(o, "y") + .35 - .20 * g(o, "height")
                a[1], a[3] = np.clip(ty - ry, -.049, .049), -.099
                if abs(ty - ry) < .012 or self.phase_step >= 35:
                    self._advance("clear_sweep")

        elif self.phase == "clear_sweep":
            o = self._named(state, self.clear_name)
            if o is None:
                self._advance("clear_select")
            else:
                ox, ow = g(o, "x"), g(o, "width")
                goal = sx + self.direction * (sw / 2.0 + ow / 2.0 + .10)
                a[0] = self.direction * .018
                if self.direction * (ox - goal) >= 0 or self.phase_step >= 75:
                    self.cleared.add(self.clear_name)
                    self._advance("clear_recover")

        elif self.phase == "clear_recover":
            a[1], a[3] = np.clip(1.12 - ry, -.049, .049), -.099
            if abs(ry - 1.12) < .012 or self.phase_step >= 30:
                self._advance("clear_select")

        elif self.phase == "raise":
            # _advance() resets to zero, then the current call increments it.
            # Thus the first action actually evaluated in this phase is step 1.
            if self.phase_step <= 1:
                self.direction = 1 if sx >= bx else -1
            a[1], a[3] = np.clip(1.10 - ry, -.049, .049), -.099
            if self.direction < 0 and (wide_low or narrow_left) and g(r, "finger_gap") < .319:
                a[4] = .019
            if abs(ry - 1.10) < .012 or self.phase_step >= 28:
                self._advance("align")
        elif self.phase == "align":
            if self.direction > 0:
                tx, want = bx - .45, 0.0
            elif narrow_left:
                tx, want = bx, -math.pi / 2.0
            elif horizontal_left:
                tx, want = min(2.76, bx + .43), math.pi
            elif wide_low:
                # A diagonal open finger reaches behind a low boundary cargo.
                tx, want = bx + bw / 2.0 - .24, -1.20
            else:
                tx, want = bx + bw / 2.0 + .10, -math.pi / 2.0
            a[0] = np.clip(tx - rx, -.049, .049)
            a[2] = np.clip(self._angle_error(want, th), -.19, .19)
            a[3] = -.099
            if (self.direction > 0 and bh < .30 and bw > .22
                    and g(r, "finger_gap") > .121):
                a[4] = -.019
            if (self.direction < 0 and not wide_low and not narrow_left
                    and g(r, "finger_gap") > .121):
                a[4] = -.019
            good = abs(tx - rx) < .012 and abs(self._angle_error(want, th)) < .025
            self.stable = self.stable + 1 if good else 0
            if self.stable >= 2 or self.phase_step >= 72:
                self._advance("lower")
        elif self.phase == "lower":
            # Contact below center so the leading bottom corner climbs the pad lip.
            if self.direction > 0:
                ty = max(.24, by - .20 * bh)
            elif narrow_left:
                ty = by + .40
            elif horizontal_left:
                ty = max(.24, by - .08)
            elif wide_low:
                ty = by + .215
            else:
                ty = by + .35 - .20 * bh
            a[1] = np.clip(ty - ry, -.049, .049)
            a[3] = -.099
            if abs(ty - ry) < .012 or self.phase_step >= 35:
                self._advance("close" if self.direction > 0 or narrow_left else "push")
        elif self.phase == "close":
            if g(r, "finger_gap") > .121:
                a[4] = -.019
            if self.phase_step >= 10:
                self._advance("push")
        else:
            left_speed = .015 if narrow_left else (.004 if horizontal_left else .005)
            a[0] = (.010 if self.direction > 0 else left_speed) * self.direction
            limit = 85 if self.direction > 0 else (180 if narrow_left else (70 if horizontal_left else 42))
            if self.phase_step >= limit:
                self._advance("raise")

        self.phase_step += 1
        return np.clip(a, self.low + 1e-7, self.high - 1e-7).astype(float)
