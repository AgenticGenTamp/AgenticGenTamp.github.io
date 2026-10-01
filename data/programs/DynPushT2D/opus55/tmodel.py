"""Quasi-static push model of the T block + geometry helpers (vectorized)."""
import numpy as np

R_ROBOT = 0.1
MU = 1.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TGeom:
    def __init__(self, w, L, lv):
        self.w, self.L, self.lv = w, L, lv
        # rects in local frame: (x0,x1,y0,y1)
        self.rects = [(-L / 2, L / 2, -w, 0.0), (-w / 2, w / 2, -w - lv, -w)]
        A1, A2 = L * w, w * lv
        cy = (A1 * (-w / 2) + A2 * (-w - lv / 2)) / (A1 + A2)
        I1 = A1 * ((L ** 2 + w ** 2) / 12 + (w / 2) ** 2)
        I2 = A2 * ((w ** 2 + lv ** 2) / 12 + (w + lv / 2) ** 2)
        self.cy = cy
        self.c2 = (I1 + I2) / (A1 + A2) - cy ** 2

    def closest(self, qx, qy):
        """Vectorized closest point on T (local frame). Returns px,py,nx,ny,dist."""
        best = None
        for (x0, x1, y0, y1) in self.rects:
            cx = np.clip(qx, x0, x1)
            cy = np.clip(qy, y0, y1)
            d = np.hypot(qx - cx, qy - cy)
            if best is None:
                best = [cx, cy, d]
            else:
                m = d < best[2]
                best[0] = np.where(m, cx, best[0])
                best[1] = np.where(m, cy, best[1])
                best[2] = np.where(m, d, best[2])
        px, py, d = best
        dd = np.maximum(d, 1e-9)
        nx = (qx - px) / dd
        ny = (qy - py) / dd
        return px, py, nx, ny, d

    def boundary_points(self, spacing=0.06):
        """Sample boundary points with outward normals (local frame), excluding
        points unreachable by the robot disc."""
        w, L, lv = self.w, self.L, self.lv
        segs = []
        # bar top: y=0, normal +y
        segs.append(((-L / 2, 0), (L / 2, 0), (0, 1)))
        # bar ends
        segs.append(((L / 2, -w), (L / 2, 0), (1, 0)))
        segs.append(((-L / 2, -w), (-L / 2, 0), (-1, 0)))
        # bar underside (both sides of stem), normal -y
        segs.append(((w / 2, -w), (L / 2, -w), (0, -1)))
        segs.append(((-L / 2, -w), (-w / 2, -w), (0, -1)))
        # stem sides
        segs.append(((w / 2, -w - lv), (w / 2, -w), (1, 0)))
        segs.append(((-w / 2, -w - lv), (-w / 2, -w), (-1, 0)))
        # stem bottom
        segs.append(((-w / 2, -w - lv), (w / 2, -w - lv), (0, -1)))
        pts = []
        for a, b, n in segs:
            a = np.array(a, float); b = np.array(b, float)
            ln = np.linalg.norm(b - a)
            k = max(2, int(np.ceil(ln / spacing)) + 1)
            for t in np.linspace(0, 1, k):
                p = a + t * (b - a)
                pts.append((p[0], p[1], n[0], n[1]))
        pts = np.array(pts)
        # robot center at contact
        cx = pts[:, 0] + pts[:, 2] * R_ROBOT
        cy = pts[:, 1] + pts[:, 3] * R_ROBOT
        _, _, _, _, d = self.closest(cx, cy)
        ok = d > R_ROBOT - 1e-6
        pts = pts[ok]
        # dedupe
        _, idx = np.unique(np.round(pts, 5), axis=0, return_index=True)
        return pts[np.sort(idx)]


GAP_K = 0.27


def _push(g, c, s, bx, by, bt, lx, ly, lux, luy):
    """Apply contact displacement (lux,luy) (local frame) at robot local pos lx,ly."""
    px, py, nx, ny, d = g.closest(lx, ly)
    c0x, c0y = 0.0, g.cy
    rrx, rry = px - c0x, py - c0y
    rpx, rpy = -rry, rrx
    c2 = g.c2
    M11 = c2 + rpx * rpx
    M12 = rpx * rpy
    M22 = c2 + rpy * rpy
    det = M11 * M22 - M12 * M12
    fx = (M22 * lux - M12 * luy) / det
    fy = (-M12 * lux + M11 * luy) / det
    tx, ty = -ny, nx
    fn = -(fx * nx + fy * ny)
    ft = fx * tx + fy * ty
    slide = np.abs(ft) > MU * fn
    sg = np.sign(ft)
    gx = -nx + sg * MU * tx
    gy = -ny + sg * MU * ty
    vpx = M11 * gx + M12 * gy
    vpy = M12 * gx + M22 * gy
    den = vpx * nx + vpy * ny
    den = np.where(np.abs(den) < 1e-12, -1e-12, den)
    lam = (lux * nx + luy * ny) / den
    fx = np.where(slide, gx * lam, fx)
    fy = np.where(slide, gy * lam, fy)
    Vx, Vy = c2 * fx, c2 * fy
    om = rrx * fy - rry * fx
    Vox = Vx + om * c0y
    Voy = Vy - om * c0x
    return bx + c * Vox - s * Voy, by + s * Vox + c * Voy, bt + om


def sim_step(g, bx, by, bt, rx, ry, ux, uy, nsub=3):
    """One env step, vectorized. Robot moves by (ux,uy) world.
    Returns new bx,by,bt,rx,ry."""
    c, s = np.cos(bt), np.sin(bt)
    lx0 = c * (rx - bx) + s * (ry - by)
    ly0 = -s * (rx - bx) + c * (ry - by)
    lux = c * ux + s * uy
    luy = -s * ux + c * uy
    px, py, nx, ny, d = g.closest(lx0, ly0)
    gap = d - R_ROBOT
    un = -(lux * nx + luy * ny)
    eff = (1.0 + GAP_K) * un - np.maximum(gap, 0.0)
    active = (un > 1e-12) & (eff > 0)
    frac = np.where(active, eff / np.maximum(un, 1e-12), 0.0)
    # contact displacement: first part (1-frac) free (or extra), rest pushes
    bx1, by1, bt1 = bx, by, bt
    if np.any(active):
        # start from robot at contact: shift robot local pos so it touches
        sx = lx0 - nx * np.maximum(gap, 0.0)
        sy = ly0 - ny * np.maximum(gap, 0.0)
        dux, duy = lux * frac / nsub, luy * frac / nsub
        cx_, cy_ = sx, sy
        for k in range(nsub):
            nbx, nby, nbt = _push(g, np.cos(bt1), np.sin(bt1), bx1, by1, bt1, cx_, cy_, dux, duy)
            # robot contact point in world after moving; re-express in new frame
            cc, ss = np.cos(bt1), np.sin(bt1)
            wx = bx1 + cc * (cx_ + dux) - ss * (cy_ + duy)
            wy = by1 + ss * (cx_ + dux) + cc * (cy_ + duy)
            cn, sn = np.cos(nbt), np.sin(nbt)
            cx_ = cn * (wx - nbx) + sn * (wy - nby)
            cy_ = -sn * (wx - nbx) + cn * (wy - nby)
            bx1 = np.where(active, nbx, bx1)
            by1 = np.where(active, nby, by1)
            bt1 = np.where(active, nbt, bt1)
    return bx1, by1, bt1, rx + ux, ry + uy


def corners_world(g, bx, by, bt):
    """All rect corners in world, arrays shape (8, K) for x and y."""
    c, s = np.cos(bt), np.sin(bt)
    xs, ys = [], []
    for (x0, x1, y0, y1) in g.rects:
        for (lx, ly) in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            xs.append(bx + c * lx - s * ly)
            ys.append(by + s * lx + c * ly)
    return np.array(xs), np.array(ys)
