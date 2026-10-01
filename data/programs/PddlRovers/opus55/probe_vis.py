import numpy as np, math
from env_client import make_env
FEAT = {"rover":["x","y","theta","store_full","calibrated","at_home"],"lander":["x","y","z"],
 "objective":["x","y","z","have_image_rover0","have_image_rover1","received_image"],
 "sample":["x","y","z","is_soil","analyzed_rover0","analyzed_rover1","received_analysis"],
 "obstacle":["x","y","z","half_x","half_y","half_z"]}
SEL = {"sample":-5/6,"calibrate":-0.5,"image":-1/6,"noop":0.0,"send":0.5,"drop":5/6}
class P:
    def __init__(s, seed=0):
        s.env = make_env(); s.reset(seed)
    def reset(s, seed=0):
        s.obs, s.info = s.env.reset(seed=seed); s.term=False; return s.obs
    def objs(s, t): return sorted(s.obs.get_objects(s.env.observation_space.get_type(t)), key=lambda o: o.name)
    def o(s, name): return s.obs.get_object_from_name(name)
    def g(s, name, f): return s.obs.get(s.o(name), f)
    def d(s, name): 
        t = name.rstrip("0123456789")
        return {f: round(s.g(name,f),3) for f in FEAT[t]}
    def dump(s):
        for t in FEAT:
            for o in s.objs(t): print(o.name, [round(s.obs.get(o,f),3) for f in FEAT[t]])
    def step(s, r=0, dx=0, dy=0, dth=0, op="noop", r2=None):
        a = np.zeros(8, dtype=np.float32)
        a[4*r:4*r+4] = [np.clip(dx,-.2,.2), np.clip(dy,-.2,.2), np.clip(dth,-.4,.4), SEL[op]]
        s.obs, s.rew, s.term, s.trunc, s.info = s.env.step(a)
        return s.obs
    def op(s, r, op): return s.step(r, op=op)
    def pose(s, r): return (s.g(f"rover{r}","x"), s.g(f"rover{r}","y"), s.g(f"rover{r}","theta"))
    def goto(s, r, x, y, th=None, tol=1e-3, maxit=60):
        for _ in range(maxit):
            cx, cy, ct = s.pose(r)
            ex, ey = x-cx, y-cy
            eth = 0 if th is None else (th-ct+math.pi)%(2*math.pi)-math.pi
            if abs(ex)<tol and abs(ey)<tol and abs(eth)<1e-3: return True
            sc = 1.0
            while True:
                n = max(1, max(abs(ex)/.2, abs(ey)/.2, abs(eth)/.4))
                s.step(r, ex/n*sc, ey/n*sc, eth/n*sc)
                nx, ny, nt = s.pose(r)
                if abs(nx-cx)>1e-7 or abs(ny-cy)>1e-7 or abs(nt-ct)>1e-7: break
                sc /= 2
                if sc < 1e-3: return False
        return False
    def turn(s, r, th):
        return s.goto(r, *s.pose(r)[:2], th=th)
    def path(s, r, pts, maxit=300):
        for (x, y) in pts:
            if not s.goto(r, x, y, maxit=maxit): return False
        return True
    def try_op(s, r, op, feat=None, obj=None):
        s.op(r, op)
        return s.g(obj or f"rover{r}", feat or "calibrated")
    # ---- simple grid A* planner (world frame) ----
    def plan(s, r, gx, gy, infl=0.24, res=0.05, goal_tol=0.0):
        import heapq
        obst = [(s.obs.get(o,'x'), s.obs.get(o,'y'), s.obs.get(o,'half_x'), s.obs.get(o,'half_y')) for o in s.objs('obstacle')]
        other = s.pose(1-r)
        def free(x, y):
            if abs(x) > 2.5-infl+0.02 or abs(y) > 2.5-infl+0.02: return False
            if abs(x) < infl: return False
            if (x > 0) != (r == 0): return False
            for (ox, oy, hx, hy) in obst:
                if abs(x-ox) < hx+infl and abs(y-oy) < hy+infl: return False
            return True
        sx, sy, _ = s.pose(r)
        N = int(round(5.0/res))+1
        idx = lambda v: int(round((v+2.5)/res))
        val = lambda i: -2.5+i*res
        start = (idx(sx), idx(sy))
        goals = set()
        gi, gj = idx(gx), idx(gy)
        R = max(0, int(goal_tol/res))
        for di in range(-R, R+1):
            for dj in range(-R, R+1):
                if math.hypot(di*res, dj*res) <= goal_tol+1e-9 and free(val(gi+di), val(gj+dj)): goals.add((gi+di, gj+dj))
        if not goals: return None
        pq = [(0, start)]; came = {start: None}; cost = {start: 0}
        while pq:
            _, cur = heapq.heappop(pq)
            if cur in goals: break
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di == dj == 0: continue
                    nb = (cur[0]+di, cur[1]+dj)
                    if not (0 <= nb[0] < N and 0 <= nb[1] < N): continue
                    if not free(val(nb[0]), val(nb[1])) and cur != start: continue
                    nc = cost[cur] + math.hypot(di, dj)
                    if nc < cost.get(nb, 1e9):
                        cost[nb] = nc; came[nb] = cur
                        h = min(math.hypot(nb[0]-g[0], nb[1]-g[1]) for g in list(goals)[:20])
                        heapq.heappush(pq, (nc+h, nb))
        else:
            return None
        pts = []
        while cur is not None: pts.append((val(cur[0]), val(cur[1]))); cur = came[cur]
        pts = pts[::-1][1:]
        # thin out
        return pts[3::4] + [pts[-1]] if pts else []
    def nav(s, r, gx, gy, **kw):
        pts = s.plan(r, gx, gy, **kw)
        if pts is None: return False
        return s.path(r, pts, maxit=60)
