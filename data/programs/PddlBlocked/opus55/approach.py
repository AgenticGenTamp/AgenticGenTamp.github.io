"""GeneratedApproach for PR2Blocked: move blocker aside, grasp green0, drop on plate."""
import time
import numpy as np
from kin import wrap, fk_world, ik
import collide
import stages
from planner import jdist, travel_configs, TABLE
from stages import geom, blocker_options, green_options, spare_options, travel_any

JN = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
MERGE_GRIP = True
LIFT_Z = 0.97
LIFT_KEEP_ORI = False
RETRY_MODE = 'legacy'
FLIP_ROLL = True
MERGE_SEGS = True
MERGE_TRAVEL = True
DEFER_TAIL = True
BF = ['pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw', 'grasp_active']


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.primitives = primitives
        self._gen = None
        self._state = None
        self.debug = False
        self._t = 0

    # ---- state access ----
    def _robot(self, s):
        try:
            return s.get_object_from_name('robot')
        except Exception:
            for t, _ in s.type_features.items():
                if getattr(t, 'name', '') == 'robot':
                    return s.get_objects(t)[0]
        return None

    def _base(self):
        s = self._state; r = self._r
        return np.array([float(s.get(r, 'base_x')), float(s.get(r, 'base_y')), float(s.get(r, 'base_rot'))])

    def _q(self):
        s = self._state; r = self._r
        return np.array([float(s.get(r, j)) for j in JN])

    def _block(self, name):
        s = self._state
        o = s.get_object_from_name(name)
        return np.array([float(s.get(o, f)) for f in BF])

    def reset(self, state, info=None):
        self._state = state
        self._r = self._robot(state)
        self._t0 = time.time()
        stages._DROP_CACHE.clear(); stages._BC_CACHE.clear()
        self._noflip = False
        self._prefix = []
        self._gen = self._run()

    def get_action(self, state):
        self._state = state
        self._r = self._robot(state) if self._r is None else self._r
        try:
            a = next(self._gen)
            self._t += 1
        except StopIteration:
            a = None
        if a is None:
            a = np.zeros(11)
        return np.asarray(a, dtype=np.float32)

    # ---- behaviour ----
    @staticmethod
    def _diff(b, q, bb, qq):
        return np.r_[bb[:2] - b[:2], wrap(bb[2] - b[2]), jdist(q, qq)]

    def _follow(self, cfgs, hist=None, final_grip=0.0):
        """Follow the polyline through cfgs taking maximal steps (inf-norm 0.2) along it."""
        self._merged = False
        if not cfgs:
            return True
        b0 = self._base(); q0 = self._q()
        V = [np.r_[b0, q0]]
        for bb, qq in cfgs:
            pb = V[-1]
            V.append(pb + self._diff(pb[:3], pb[3:], np.asarray(bb, float), np.asarray(qq, float)))
        V = np.array(V)
        N = len(V) - 1
        s = 0.0
        LIM = 0.2 - 1e-6

        def P(x):
            k = min(int(np.floor(x)), N - 1)
            t = x - k
            return V[k] + t * (V[k + 1] - V[k])
        while True:
            b = self._base(); q = self._q()
            cur = np.r_[b, q]

            def dist(x):
                p = P(x)
                return np.abs(self._diff(b, q, p[:3], p[3:])).max()
            k = int(np.floor(s)) + 1
            s_new = s
            while k <= N:
                if dist(k) <= LIM:
                    s_new = float(k); k += 1
                    continue
                lo, hi = s_new, float(k)
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if dist(mid) <= LIM:
                        lo = mid
                    else:
                        hi = mid
                s_new = lo
                break
            p = P(s_new)
            delta = self._diff(b, q, p[:3], p[3:])
            m = np.abs(delta).max()
            last = s_new >= N - 1e-6
            if m < 1e-4:
                if last:
                    return True
                s = float(min(N, int(np.floor(s_new)) + 1))
                continue
            a = np.r_[delta, final_grip if (last and MERGE_GRIP) else 0.0]
            yield a
            now = np.r_[self._base(), self._q()]
            if np.abs(now - cur).max() < 1e-6:
                self._log('BLOCKED', b.round(3), q.round(2))
                return False
            if hist is not None:
                hist.append((self._base(), self._q()))
            if last and MERGE_GRIP and final_grip != 0:
                self._merged = True
                return True
            s = s_new

    def _late(self, limit=40.0):
        return time.time() - self._t0 > limit

    def _log(self, *a):
        if self.debug:
            print('[%d]' % self._t, *a, flush=True)

    def _grip(self, g):
        yield np.r_[np.zeros(10), g]

    def _held(self):
        for n in self._names:
            try:
                if self._block(n)[7] > 0.5:
                    return n
            except Exception:
                pass
        return None

    def _moved(self, prev):
        return np.abs(np.r_[self._base(), self._q()] - prev).max() > 1e-6

    def _unstick(self, tries=3):
        """Try small escape moves until the robot moves. Returns True if it moved."""
        moved_any = False
        for _ in range(tries):
            b = self._base()
            away = b[:2] - np.array(TABLE[:2] if b[0] > 0 else (-4.5, 0.0))
            away = away / (np.linalg.norm(away) + 1e-9)
            cands = []
            for k, dv in ((1, -0.1), (1, -0.2), (3, 0.1), (3, -0.1), (0, 0.1), (0, -0.1), (2, 0.1), (2, -0.1)):
                a = np.zeros(11); a[3 + k] = dv; cands.append(a)
            a = np.zeros(11); a[:2] = 0.1 * away; cands.append(a)
            a = np.zeros(11); a[:2] = 0.1 * away; a[4] = -0.1; cands.append(a)
            ok = False
            for a in cands:
                prev = np.r_[self._base(), self._q()]
                yield a
                if self._moved(prev):
                    ok = True
                    break
            if not ok:
                return moved_any
            moved_any = True
        return moved_any

    def _lift_safe(self, zmin=None):
        """Raise the tool vertically (fixed base) if it is low."""
        zmin = LIFT_Z if zmin is None else zmin
        b = self._base(); q = self._q()
        p, R = fk_world(b, q)
        if p[2] >= zmin - 1e-3:
            return True
        d = R[:, 0].copy(); d[2] = 0.0
        if np.linalg.norm(d) < 1e-3:
            return True
        d /= np.linalg.norm(d)
        cfgs = []
        qq = q
        n = int(np.ceil((zmin - p[2]) / 0.03))
        for i in range(1, n + 1):
            tgt = p + np.array([0, 0, (zmin - p[2]) * i / n])
            if LIFT_KEEP_ORI:
                _, qn, err = ik(tgt, R[:, 0], b, qq, free_base=False, n_restarts=0, zaxis=R[:, 2])
            else:
                _, qn, err = ik(tgt, d, b, qq, free_base=False, n_restarts=0)
            if err > 1e-2 or np.abs(jdist(qq, qn)).max() > 0.5:
                break
            qq = qn
            cfgs.append((b, qq))
        if not cfgs:
            return False
        ok = yield from self._follow(cfgs)
        return ok

    def _exec(self, segs, check_grasp=None):
        """Travel to segs[0] then run all segments. On blockage, release and back out.
        Returns True on completion (or env done)."""
        if FLIP_ROLL and not getattr(self, '_noflip', False):
            q6now = self._prefix[-1][1][6] if self._prefix else self._q()[6]
            dl = wrap(segs[0]['configs'][-1][1][6] - q6now)
            if abs(dl) > np.pi / 2:
                sh = -np.pi if dl > 0 else np.pi
                out = []
                for w in segs:
                    if w['name'] in ('carry', 's_carry') and out:
                        pb, pq = out[-1]['configs'][-1]
                        eb, eq = w['configs'][-1]
                        out.append(dict(w, configs=travel_any(pb, pq, eb, eq)))
                    else:
                        out.append(dict(w, configs=[(bb, np.r_[qq[:6], wrap(qq[6] + sh)])
                                                    for bb, qq in w['configs']]))
                segs = out
        b0, q0 = segs[0]['configs'][-1]
        prefix = self._prefix
        self._prefix = []
        if prefix:
            sb, sq = prefix[-1]
        else:
            ok = yield from self._lift_safe()
            if not ok:
                moved = yield from self._unstick()
                if not moved:
                    return False
                yield from self._lift_safe()
            sb, sq = self._base(), self._q()
        rest = segs[1:]
        if MERGE_SEGS:
            m = []
            carry_over = []
            for w in rest:
                if w['grip'] == 0 and w['name'] not in ('carry', 's_carry', 'carry_lift'):
                    carry_over += list(w['configs'])
                    continue
                m.append(dict(w, configs=carry_over + list(w['configs'])))
                carry_over = []
            if carry_over:
                m.append(dict(name='tail', configs=carry_over, grip=0.0))
            rest = m
        tc = list(prefix) + list(travel_any(sb, sq, b0, q0))
        first_done = False
        if MERGE_TRAVEL and rest and rest[0]['name'] not in ('carry', 's_carry', 'carry_lift'):
            w0 = rest[0]
            ok = yield from self._follow(list(tc) + list(w0['configs']), None, w0['grip'])
            if ok:
                first_done = True
                if w0['grip'] != 0 and not self._merged:
                    yield from self._grip(w0['grip'])
                if w0['grip'] < 0 and self._held() is None:
                    self._log('  merged grasp failed', w0['name'])
                    yield from self._grip(1.0)
                    return False
        else:
            ok = yield from self._follow(tc)
        self._log('  travel', ok, first_done)
        if not ok and prefix:
            yield from self._lift_safe()
        if not ok:
            moved = yield from self._unstick()
            if not moved:
                return False
            import planner as _pl
            _pl.TRAVEL_MODE = RETRY_MODE
            try:
                tc2 = travel_any(self._base(), self._q(), b0, q0)
            finally:
                _pl.TRAVEL_MODE = 'spread'
            ok = yield from self._follow(tc2)
            if not ok:
                return False
        if first_done:
            hist = [(np.asarray(b0, float), np.asarray(q0, float)), (self._base(), self._q())]
            rest = rest[1:]
        else:
            hist = [(self._base(), self._q())]
        for w in rest:
            if DEFER_TAIL and w is rest[-1] and w['name'] == 'tail':
                self._prefix = list(w['configs'])
                break
            ok = yield from self._follow(w['configs'], hist, w['grip'])
            self._log('  seg', w['name'], ok)
            if ok and w['grip'] != 0:
                if not self._merged:
                    yield from self._grip(w['grip'])
                if w['grip'] < 0 and self._held() is None:
                    ok = False
            if not ok and w['name'] in ('carry', 's_carry') and self._held() is not None:
                ok = yield from self._recover_carry(w)
                self._log('  carry recover', ok)
            if not ok:
                self._log('  fail in seg', w['name'], 'held', self._held())
                if self._held() is not None:
                    yield from self._grip(1.0)
                ok = yield from self._follow(list(reversed(hist)))
                if not ok:
                    yield from self._unstick()
                return False
        return True

    def _recover_carry(self, w):
        import planner as _pl
        eb, eq = w['configs'][-1]
        for mode in (RETRY_MODE, 'tuck'):
            moved = yield from self._unstick()
            if not moved:
                return False
            yield from self._lift_safe()
            _pl.TRAVEL_MODE = mode
            try:
                tc = travel_any(self._base(), self._q(), eb, eq)
            finally:
                _pl.TRAVEL_MODE = 'spread'
            ok = yield from self._follow(tc)
            if ok and self._held() is not None:
                if w['grip'] != 0:
                    yield from self._grip(w['grip'])
                return True
            if self._held() is None:
                return False
        return False

    def _blocker_in_way(self, g0, d):
        blk = self._block('blocker')
        v = blk[:2] - g0[:2]
        along = v @ d[:2]
        lat = abs(v @ np.array([-d[1], d[0]]))
        return (-0.40 < along < 0.0) and lat < 0.10

    def _run(self):
        self._names = [n for n in self._state.get_object_names() if n == 'blocker' or n.startswith('green')]
        blk = self._block('blocker'); g0 = self._block('green0')
        d, perp, z = geom(blk, g0, g0[2])
        self._pen = collide.pen_boxes(g0[0], g0[1], self._yaw(g0))
        self._set_obs(skip={'blocker'})
        for name, segs in stages.blocker_options_sorted(self._base(), self._q(), blk, g0, g0[2],
                                                              deadline=self._t0 + 12.0):
            if not self._blocker_in_way(g0, d) or self._late():
                break
            ok = yield from self._exec(segs)
            self._log('blocker opt', name, ok)
        blk = self._block('blocker')
        self._set_obs(skip={'green0'})
        for name, segs in green_options(self._base(), g0, blk, d, g0[2], cur_q=self._q(),
                                        deadline=self._t0 + 25.0):
            if self._late():
                break
            ok = yield from self._exec(segs)
            self._log('green opt', name, ok)
            self._noflip = True
            if ok:
                return
        yield from self._spares()

    def _set_obs(self, skip=()):
        boxes = dict(self._pen)
        for n in self._names:
            bb = self._block(n)
            if bb[0] > 0:
                boxes[n] = (bb[0], bb[1], bb[2], 0.035, 0.035, 0.07, self._yaw(bb))
        collide.set_obstacles(boxes)
        collide.CTX['skip'] = set(skip)

    def _yaw(self, b):
        x, y, z, w = b[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _spares(self):
        while True:
            sp = []
            for n in self._names:
                if n.startswith('green') and n != 'green0':
                    b = self._block(n)
                    if b[0] < 0:
                        sp.append((b[0], b[1], self._yaw(b)))
            if not sp:
                return
            any_opt = False
            if self._late():
                return
            self._set_obs(skip=set(self._names))
            for name, segs in spare_options(self._base(), sp, 0.8):
                any_opt = True
                ok = yield from self._exec(segs)
                self._log('spare opt', name, ok)
                if ok:
                    return
                break
            if not any_opt:
                return
