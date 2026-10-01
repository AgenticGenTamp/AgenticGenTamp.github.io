"""Approach for the PDDLStream 'rovers' environment (2 rovers, variable objectives).

Plan: build a geometric model of the arena from the object-centric state,
compute candidate stand regions for each operator, assign tasks to the two
rovers (they are permanently separated by the middle wall), order them with a
small DP, then execute with a greedy step-wise controller that also applies
operators opportunistically while driving.
"""
import itertools
import numpy as np

import geom
import planner

OP_SAMPLE = -1.0 + 0.5 / 3
OP_CALIB = -1.0 + 1.5 / 3
OP_IMAGE = -1.0 + 2.5 / 3
OP_NOOP = -1.0 + 3.5 / 3
OP_SEND = -1.0 + 4.5 / 3
OP_DROP = -1.0 + 5.5 / 3

STEP = geom.STEP
HOME_TOL = 0.22      # at_home radius is 0.25 (Euclidean)


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rover_xy(state, k):
        o = state.get_object_from_name("rover%d" % k)
        return np.array([float(state.get(o, "x")), float(state.get(o, "y"))])

    @staticmethod
    def _feat(state, name, feat):
        return float(state.get(state.get_object_from_name(name), feat))

    @staticmethod
    def _names(state, prefix):
        return sorted([n for n in state.get_object_names() if n.startswith(prefix)],
                      key=lambda s: int(s[len(prefix):]))

    @staticmethod
    def _xy(state, name):
        o = state.get_object_from_name(name)
        return np.array([float(state.get(o, "x")), float(state.get(o, "y"))])

    # ------------------------------------------------------------------
    def reset(self, state, info=None):
        self.model = geom.Model(state)
        self.graph = planner.GridGraph(self.model, clearance=0.01)
        self.home = [self._rover_xy(state, 0), self._rover_xy(state, 1)]
        self.blocked = set()      # grid nodes learned to be unreachable
        self.last_pos = [None, None]
        self.last_cmd = [None, None]
        self.stuck = [0, 0]
        self.detour = [None, None]
        self.opfail = [0, 0]
        self.rng = np.random.default_rng(0)
        self.escape = [0, 0]
        self.prev = [None, None]
        self._compute_regions(state)
        self._assign(state)
        return None

    # ------------------------------------------------------------------
    def _compute_regions(self, state):
        g, m = self.graph, self.model
        self.objs = self._names(state, "objective")
        self.samps = self._names(state, "sample")
        self.obj_xy = {n: self._xy(state, n) for n in self.objs}
        self.samp_xy = {n: self._xy(state, n) for n in self.samps}
        self.soil = {n: self._feat(state, n, "is_soil") > 0.5 for n in self.samps}
        self.home_nodes = [g.node_of(self.home[k]) for k in range(2)]
        self.d_home, self.p_home = g.dists(self.home_nodes)
        reach_any = ((np.isfinite(self.d_home[0]) & (self.d_home[0] < 1e8)) |
                     (np.isfinite(self.d_home[1]) & (self.d_home[1] < 1e8)))
        self.view_mask = {}
        for n in self.objs:
            for (rng, pad) in ((geom.IMG_RANGE, geom.SIGHT_PAD), (1.9, 0.10),
                               (1.98, 0.02), (1.98, -0.04)):
                mk = planner.objective_view_mask(m, g, self.obj_xy[n], rng, pad)
                if np.any(mk & reach_any):
                    break
            self.view_mask[n] = mk
        self.samp_maskd = {}
        for n in self.samps:
            mk = planner.sample_mask(g, self.samp_xy[n])
            if not np.any(mk & reach_any):
                mk = planner.sample_mask(g, self.samp_xy[n], 0.245)
            self.samp_maskd[n] = mk
        self.send_maskd = planner.send_mask(m, g)
        anchors = [self.home[0], self.home[1], m.lander]
        reps = {}
        for n in self.objs:
            reps[("obj", n)] = planner.pick_reps(
                g, np.nonzero(self.view_mask[n])[0], 6, anchors)
        for n in self.samps:
            reps[("samp", n)] = planner.pick_reps(
                g, np.nonzero(self.samp_maskd[n])[0], 4, anchors)
        reps[("send", None)] = planner.pick_reps(
            g, np.nonzero(self.send_maskd)[0], 8, anchors)
        self.home_reps = []
        for k in range(2):
            dh = np.hypot(g.pts[:, 0] - self.home[k][0], g.pts[:, 1] - self.home[k][1])
            nodes = np.nonzero(dh <= HOME_TOL)[0]
            sendable = [int(n) for n in nodes if self.send_maskd[n]]
            base = [int(nodes[int(np.argmin(dh[nodes]))])] if len(nodes) else [self.home_nodes[k]]
            hr = base + planner.pick_reps(g, np.array(sendable), 4, [self.home[k]]) \
                if sendable else base
            seen, hh = set(), []
            for n in hr:
                if n not in seen:
                    seen.add(n)
                    hh.append(n)
            self.home_reps.append(hh)
            reps[("send", None)] = list(dict.fromkeys(
                reps[("send", None)] + [n for n in hh if self.send_maskd[n]]))
        reps[("home0", None)] = self.home_reps[0]
        reps[("home1", None)] = self.home_reps[1]
        self.reps = reps
        src = sorted(set(self.home_nodes) | {r for v in reps.values() for r in v})
        self.src = src
        self.src_index = {s: i for i, s in enumerate(src)}
        self.D, self.P = g.dists(src)

    def _reach(self, k, node):
        return np.isfinite(self.d_home[k][node]) and self.d_home[k][node] < 1e8

    def _dist(self, a, b):
        d = self.D[self.src_index[a]][b]
        return d if np.isfinite(d) else np.inf

    # ------------------------------------------------------------------
    def _assign(self, state):
        objs, samps = self.objs, self.samps
        stones = [n for n in samps if not self.soil[n]]
        soils = [n for n in samps if self.soil[n]]
        self._cost_memo = {}
        nobj = len(objs)
        # objective assignment is forced by reachability, but enumerate anyway
        best = None
        for stone in stones:
            for soil in soils:
                for sr in (0, 1):
                    for lr in (0, 1):
                        for bits in range(1 << nobj):
                            sets = [[objs[i] for i in range(nobj) if not (bits >> i) & 1],
                                    [objs[i] for i in range(nobj) if (bits >> i) & 1]]
                            sm = [[], []]
                            sm[sr].append(stone)
                            sm[lr].append(soil)
                            c0, r0 = self._rover_cost(0, sets[0], sm[0])
                            c1, r1 = self._rover_cost(1, sets[1], sm[1])
                            if not (np.isfinite(c0) and np.isfinite(c1)):
                                continue
                            cost = max(c0, c1) + 0.02 * min(c0, c1)
                            if best is None or cost < best[0]:
                                best = (cost, [r0, r1])
        if best is None:
            best = (0.0, [self._rover_cost(k, [], [])[1] for k in range(2)])
        self.tasks = [list(r) for r in best[1]]

    def _rover_cost(self, k, objsub, sampsub):
        key = (k, tuple(sorted(objsub)), tuple(sorted(sampsub)))
        if key not in self._cost_memo:
            self._cost_memo[key] = self._rover_cost_raw(k, objsub, sampsub)
        return self._cost_memo[key]

    def _rover_cost_raw(self, k, objsub, sampsub):
        items = []
        dropped = 0
        for key in [("samp", n) for n in sampsub] + [("obj", n) for n in objsub]:
            reach = [r for r in self.reps.get(key, []) if self._reach(k, r)]
            if reach:
                items.append(key)
            else:
                dropped += 1
        hn = self.home_nodes[k]
        best = None
        for order in itertools.permutations(range(len(items))):
            cur = {hn: (0.0, [])}
            ok = True
            for oi in order:
                key = items[oi]
                nxt = {}
                for rep in [r for r in self.reps[key] if self._reach(k, r)]:
                    bv = None
                    for node, (c, seq) in cur.items():
                        d = self._dist(node, rep)
                        if not np.isfinite(d):
                            continue
                        v = c + d
                        if bv is None or v < bv[0]:
                            bv = (v, seq + [(key, rep)])
                    if bv is not None:
                        nxt[rep] = bv
                if not nxt:
                    ok = False
                    break
                cur = nxt
            if not ok:
                continue
            fin = None
            homes = [h for h in self.home_reps[k] if self._reach(k, h)] or [hn]
            if items:
                for rep in self.reps[("send", None)]:
                    if not self._reach(k, rep):
                        continue
                    for hnode in homes:
                        d2 = self._dist(rep, hnode)
                        if not np.isfinite(d2):
                            continue
                        for node, (c, seq) in cur.items():
                            d1 = self._dist(node, rep)
                            if not np.isfinite(d1):
                                continue
                            v = c + d1 + d2
                            if fin is None or v < fin[0]:
                                fin = (v, seq + [(("send", None), rep),
                                                 (("home", None), hnode)])
            else:
                for hnode in homes:
                    for node, (c, seq) in cur.items():
                        v = c + self._dist(node, hnode)
                        if fin is None or v < fin[0]:
                            fin = (v, seq + [(("home", None), hnode)])
            if fin is None:
                continue
            if best is None or fin[0] < best[0]:
                best = fin
        if best is None:
            return np.inf, None
        steps = best[0] * self.graph.res / STEP
        cost = steps + 2.0 * len(objsub) + 1.0 * len(sampsub) + 500.0 * dropped
        tasks = [{"kind": key[0], "obj": key[1], "node": node} for key, node in best[1]]
        return cost, tasks

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def get_action(self, state):
        act = np.zeros(8, dtype=np.float32)
        pos = [self._rover_xy(state, 0), self._rover_xy(state, 1)]
        # detect rejected motions -> learn blocked cells
        for k in range(2):
            if self.last_cmd[k] is not None and self.last_pos[k] is not None:
                cmd = self.last_cmd[k]
                if (abs(cmd[0]) > 1e-6 or abs(cmd[1]) > 1e-6) and \
                        np.allclose(pos[k], self.last_pos[k], atol=1e-6):
                    tgt = self.last_pos[k] + np.array(cmd[:2])
                    node = self.graph.node_of(tgt)
                    self.blocked.add(node)
                    self.stuck[k] += 1
                else:
                    self.stuck[k] = 0
        # detect failed operators and adapt the stand point
        for k in range(2):
            pv = self.prev[k]
            snap = self._snap(state, k)
            if pv is not None:
                op, osnap = pv
                failed = False
                if op == OP_CALIB and snap[0] <= osnap[0]:
                    failed = True
                elif op == OP_IMAGE and snap[1] <= osnap[1]:
                    failed = True
                elif op == OP_SAMPLE and snap[2] <= osnap[2]:
                    failed = True
                elif op == OP_SEND and snap[3] <= osnap[3]:
                    failed = True
                if failed:
                    self.opfail[k] += 1
                    self._adapt(k, pos[k], op)
                else:
                    self.opfail[k] = 0
                if self.tasks[k] and self.stuck[k] >= 6:
                    self.opfail[k] += 1
                    self.stuck[k] = 0
                    self._adapt(k, pos[k], OP_NOOP)
            self.prev[k] = (None, snap)
        for k in range(2):
            dx, dy, op = self._rover_action(state, k, pos)
            self.prev[k] = (op, self.prev[k][1])
            act[4 * k + 0] = dx
            act[4 * k + 1] = dy
            act[4 * k + 3] = op
            self.last_cmd[k] = (dx, dy, op)
            self.last_pos[k] = pos[k].copy()
        return act

    def _snap(self, state, k):
        cal = self._feat(state, "rover%d" % k, "calibrated")
        nimg = sum(self._feat(state, n, "have_image_rover%d" % k) for n in self.objs)
        full = self._feat(state, "rover%d" % k, "store_full")
        recv = sum(self._feat(state, n, "received_image") for n in self.objs) + \
            sum(self._feat(state, n, "received_analysis") for n in self.samps)
        return (cal, nimg, full, recv)

    def _adapt(self, k, p, op):
        """An operator was refused: pick a different stand point for this task."""
        if not self.tasks[k]:
            return
        t = self.tasks[k][0]
        g = self.graph
        reach = np.isfinite(self.d_home[k]) & (self.d_home[k] < 1e8)
        if t["kind"] == "obj":
            xy = self.obj_xy[t["obj"]]
            d = np.hypot(g.pts[:, 0] - xy[0], g.pts[:, 1] - xy[1])
            lim = max(0.35, geom.IMG_RANGE - 0.4 * self.opfail[k])
            cand = np.nonzero(reach & (d <= lim))[0]
            if len(cand):
                dc = np.hypot(g.pts[cand, 0] - p[0], g.pts[cand, 1] - p[1])
                t["node"] = int(cand[int(np.argmin(dc + 0.3 * d[cand]))])
        elif t["kind"] == "send":
            pts = g.pts
            dl = np.hypot(pts[:, 0] - self.model.lander[0], pts[:, 1] - self.model.lander[1])
            band = (geom.WALL_BAND[0] - 0.12 * self.opfail[k],
                    geom.WALL_BAND[1] + 0.12 * self.opfail[k])
            old = geom.WALL_BAND
            geom.WALL_BAND = band
            try:
                ok = self.model.can_send_many(pts)
            finally:
                geom.WALL_BAND = old
            cand = np.nonzero(reach & ok & (dl <= geom.COMM_RANGE - 0.3))[0]
            if len(cand):
                dc = np.hypot(pts[cand, 0] - p[0], pts[cand, 1] - p[1])
                t["node"] = int(cand[int(np.argmax(-dc))])
        elif t["kind"] == "samp" and self.opfail[k] >= 3:
            self._switch_sample(k, p)
        elif t["kind"] == "samp" and op == OP_SAMPLE:
            xy = self.samp_xy[t["obj"]]
            d = np.hypot(g.pts[:, 0] - xy[0], g.pts[:, 1] - xy[1])
            cand = np.nonzero(reach & (d <= max(0.05, 0.22 - 0.05 * self.opfail[k])))[0]
            if len(cand):
                dc = np.hypot(g.pts[cand, 0] - p[0], g.pts[cand, 1] - p[1])
                t["node"] = int(cand[int(np.argmin(dc))])

    def _switch_sample(self, k, p):
        """Give up on the current sample; use another one of the same kind."""
        t = self.tasks[k][0]
        g = self.graph
        reach = np.isfinite(self.d_home[k]) & (self.d_home[k] < 1e8)
        want = self.soil[t["obj"]]
        tried = getattr(self, "_tried_samp", set())
        tried.add((k, t["obj"]))
        self._tried_samp = tried
        best = None
        for n in self.samps:
            if self.soil[n] != want or (k, n) in tried:
                continue
            cand = np.nonzero(reach & self.samp_maskd[n])[0]
            if not len(cand):
                continue
            dc = np.hypot(g.pts[cand, 0] - p[0], g.pts[cand, 1] - p[1])
            i = int(np.argmin(dc))
            if best is None or dc[i] < best[0]:
                best = (dc[i], n, int(cand[i]))
        if best is not None:
            t["obj"] = best[1]
            t["node"] = best[2]
            self.opfail[k] = 0
        else:
            # hand the task to the other rover
            o = 1 - k
            reach_o = np.isfinite(self.d_home[o]) & (self.d_home[o] < 1e8)
            for n in self.samps:
                if self.soil[n] != want:
                    continue
                cand = np.nonzero(reach_o & self.samp_maskd[n])[0]
                if len(cand):
                    self.tasks[o].insert(0, {"kind": "samp", "obj": n,
                                             "node": int(cand[0])})
                    break
            self.tasks[k].pop(0)
            self.opfail[k] = 0

    def _prune(self, state, k):
        while self.tasks[k]:
            t = self.tasks[k][0]
            if t["kind"] == "obj":
                if self._feat(state, t["obj"], "have_image_rover%d" % k) > 0.5 or \
                        self._feat(state, t["obj"], "received_image") > 0.5:
                    self.tasks[k].pop(0)
                    continue
            elif t["kind"] == "samp":
                if self._feat(state, t["obj"], "analyzed_rover%d" % k) > 0.5 or \
                        self._sample_type_done(state, self.soil[t["obj"]], k):
                    self.tasks[k].pop(0)
                    continue
            elif t["kind"] == "send":
                if not self._pending(state, k):
                    self.tasks[k].pop(0)
                    continue
            break

    def _sample_type_done(self, state, is_soil, k):
        """Another sample of this type already analysed-or-received for this rover."""
        for n in self.samps:
            if self.soil[n] != is_soil:
                continue
            if self._feat(state, n, "received_analysis") > 0.5:
                return True
            if self._feat(state, n, "analyzed_rover%d" % k) > 0.5:
                return True
        return False

    def _pending(self, state, k):
        out = []
        for n in self.objs:
            if self._feat(state, n, "have_image_rover%d" % k) > 0.5 and \
                    self._feat(state, n, "received_image") < 0.5:
                out.append(n)
        for n in self.samps:
            if self._feat(state, n, "analyzed_rover%d" % k) > 0.5 and \
                    self._feat(state, n, "received_analysis") < 0.5:
                out.append(n)
        return out

    # ------------------------------------------------------------------
    def _rover_action(self, state, k, pos):
        self._prune(state, k)
        p = pos[k]
        store_full = self._feat(state, "rover%d" % k, "store_full") > 0.5
        calibrated = self._feat(state, "rover%d" % k, "calibrated") > 0.5

        if not self.tasks[k]:
            at_home = np.linalg.norm(p - self.home[k]) <= HOME_TOL
            op = OP_DROP if store_full else OP_NOOP
            if at_home:
                return 0.0, 0.0, op
            d = self._move(k, p, self.home_nodes[k])
            return d[0], d[1], op

        t = self.tasks[k][0]
        node = t["node"]
        op = OP_NOOP

        if t["kind"] == "samp":
            name = t["obj"]
            if np.linalg.norm(p - self.samp_xy[name]) <= geom.SAMPLE_RANGE - 0.005 \
                    and self.opfail[k] < 3:
                return 0.0, 0.0, OP_DROP if store_full else OP_SAMPLE
            if store_full:
                op = OP_DROP
        elif t["kind"] == "obj":
            name = t["obj"]
            tgt = self._image_target(state, k, p)
            if tgt is not None and self._feat(state, tgt, "received_image") < 0.5:
                return 0.0, 0.0, OP_IMAGE if calibrated else OP_CALIB
        elif t["kind"] == "send":
            if self._can_send(p):
                return 0.0, 0.0, OP_SEND
        elif t["kind"] == "home":
            if np.linalg.norm(p - self.home[k]) <= HOME_TOL:
                self.tasks[k].pop(0)
                return 0.0, 0.0, OP_DROP if store_full else OP_NOOP

        if op == OP_NOOP and store_full and t["kind"] != "samp":
            op = OP_DROP
        d = self._move(k, p, node)
        return d[0], d[1], op

    def _image_target(self, state, k, p):
        """Objective the env would photograph from p: nearest visible one still needed."""
        best, bd = None, np.inf
        for n in self.objs:
            if self._feat(state, n, "have_image_rover%d" % k) > 0.5:
                continue
            if not self._can_image(p, n):
                continue
            d = np.linalg.norm(p - self.obj_xy[n])
            if d < bd:
                best, bd = n, d
        return best

    def _can_image(self, p, name):
        xy = self.obj_xy[name]
        if np.linalg.norm(p - xy) > geom.IMG_RANGE:
            return False
        return self.model.visible(p, xy)

    def _can_send(self, p):
        return self.model.can_send(p)

    # ------------------------------------------------------------------
    def _move(self, k, p, node):
        """One step (dx, dy) toward grid node `node`, respecting learned blocks."""
        g = self.graph
        start = g.node_of(p)
        pts = self._path(start, node)
        if pts is None or len(pts) == 0:
            v = g.pts[node] - p
            if max(abs(v[0]), abs(v[1])) < 1e-6:
                return 0.0, 0.0
            v = v / max(abs(v[0]), abs(v[1])) * STEP
            ang = 0.9 * (self.stuck[k] + 1)
            c, s2 = np.cos(ang), np.sin(ang)
            v = np.array([c * v[0] - s2 * v[1], s2 * v[0] + c * v[1]])
            return float(np.clip(v[0], -STEP, STEP)), float(np.clip(v[1], -STEP, STEP))
        target = None
        for i in range(len(pts) - 1, -1, -1):
            q = pts[i]
            if max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= STEP + 1e-9 and \
                    self._seg_free(p, q):
                target = q
                break
        if target is None:
            target = pts[min(1, len(pts) - 1)]
            if max(abs(target[0] - p[0]), abs(target[1] - p[1])) < 1e-6 and len(pts) > 1:
                target = pts[-1]
        if self.escape[k] > 0 or self.stuck[k] >= 3:
            if self.escape[k] <= 0:
                self.escape[k] = 4
            self.escape[k] -= 1
            v = self.rng.uniform(-STEP, STEP, size=2)
            return float(v[0]), float(v[1])
        if self.stuck[k] >= 2:
            # nudge sideways to escape
            ang = 1.0 + 0.7 * self.stuck[k]
            v = target - p
            c, s = np.cos(ang), np.sin(ang)
            v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                v = v / max(abs(v[0]), abs(v[1])) * STEP
            target = p + v
        d = np.clip(target - p, -STEP, STEP)
        return float(d[0]), float(d[1])

    def _seg_free(self, p, q):
        n = max(2, int(np.ceil(np.linalg.norm(q - p) / 0.05)) + 1)
        ts = np.linspace(0.0, 1.0, n)
        pts = p[None, :] + ts[:, None] * (q - p)[None, :]
        return bool(np.all(self.model.free_points(pts)))

    def _path(self, start, node):
        g = self.graph
        if not self.blocked and node in self.src_index:
            pred = self.P[self.src_index[node]]
            if pred[start] < 0 and start != node:
                return None
            path = g.path(pred, start)
            return g.pts[path][::-1]
        # replan with learned blocks removed
        return self._path_dyn(start, node)

    def _path_dyn(self, start, node):
        g = self.graph
        key = ("dyn", node, len(self.blocked))
        cache = getattr(self, "_dyn_cache", {})
        if key not in cache:
            graph = g.graph
            if self.blocked:
                graph = graph.copy()
                bl = np.array(sorted(self.blocked))
                mask = np.ones(graph.shape[0], dtype=bool)
                mask[bl] = False
                keep = mask[graph.indices]
                graph.data = graph.data * keep
                graph.eliminate_zeros()
            from scipy.sparse.csgraph import dijkstra
            d, pred = dijkstra(graph, indices=[node], unweighted=True,
                               return_predecessors=True, directed=False)
            cache[key] = pred[0]
            self._dyn_cache = cache
        pred = cache[key]
        if pred[start] < 0 and start != node:
            return None
        path = g.path(pred, start)
        return g.pts[path][::-1]
