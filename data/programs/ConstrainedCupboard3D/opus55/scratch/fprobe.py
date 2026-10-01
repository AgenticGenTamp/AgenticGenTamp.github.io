"""Fingertip probing: closed gripper pointing +x (ee z=+x). usage: seed k(n) BX 'y1,..' 'z1,..' [xstart xend]"""
import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *
QH = [np.array([0, 2.1, np.pi, -1.2, 0, 1.7, np.pi/2]), np.array([0, 1.4, np.pi, -1.9, 0, 1.8, np.pi/2]),
      np.array([0, 1.5, np.pi, -1.6, 0, 1.5, np.pi/2])]
TIP = 0.01
def ikh(S, pw, R=RH, b=None):
    b = S.base() if b is None else np.asarray(b, float)
    pa = world_to_arm(b, np.asarray(pw, float)); Ra = R_world_to_arm(b, R); best = None
    for s0 in [S.q()] + QH:
        q, e1, e2 = ik(s0, pa, Ra, tool=TOOL, iters=300)
        if e1 < 1e-3 and e2 < 1e-2 and q[1] > 0.5:
            d = q - S.q(); d[[0,2,4,6]] = (d[[0,2,4,6]] + np.pi) % (2*np.pi) - np.pi; c = np.abs(d).sum()
            if best is None or c < best[0]: best = (c, q)
    return None if best is None else best[1]

def fpush(p, start, direction, dist, v=0.002, thr=0.006):
    S = p.S; start = np.asarray(start, float); direction = np.asarray(direction, float); b = S.base()
    qt = ikh(S, start)
    if qt is None: return 'noik'
    goto_slow(S, qt=qt, steps=300, vmax=0.06)
    g0 = grasp_point_world(S)
    if np.linalg.norm(g0 - start) > 0.01: return ('startfail', g0.round(3).tolist())
    s = 0.0; res = None
    while s < dist:
        s += v; tgt = start + s*direction
        qt, e1, _ = ik(S.q(), world_to_arm(b, tgt), R_world_to_arm(b, RH), tool=TOOL)
        e = qt - S.q(); e[[0,2,4,6]] = (e[[0,2,4,6]] + np.pi) % (2*np.pi) - np.pi
        a = np.zeros(11); a[10] = S.grip; a[3:10] = np.clip(e, -0.1, 0.1); S.step(a)
        g = grasp_point_world(S)
        if np.linalg.norm(g - tgt) > thr:
            res = g.round(4); break
    p.chk()
    qt = ikh(S, start); goto_slow(S, qt=qt, steps=150, vmax=0.05)
    return res

def bpush(p, y, z, gx0, gx1, reach=0.62, v=0.002, thr=0.005, R=None):
    """arm fixed in horizontal pose (grasp point 'reach' ahead of mount); move base along +x."""
    S = p.S; bx0 = gx0 - reach - MX
    b0 = np.array([bx0, y, 0.])
    qt = ikh(S, [gx0, y, z], b=b0) if R is None else ik_multi(S, [gx0, y, z], R, b=b0)
    if qt is None: return 'noik'
    goto_slow(S, qt=qt, bt=b0, steps=300, vmax=0.06)
    g0 = grasp_point_world(S)
    if np.linalg.norm(g0 - [gx0, y, z]) > 0.01: return ('startfail', g0.round(3).tolist())
    s = 0.0; res = None
    while gx0 + s < gx1:
        s += v; bt = b0 + [s, 0, 0]
        a = np.zeros(11); a[10] = S.grip
        e = qt - S.q(); a[3:10] = np.clip(e, -0.1, 0.1)
        a[0:3] = np.clip(bt - S.base(), -0.1, 0.1); S.step(a)
        g = grasp_point_world(S)
        if np.linalg.norm(g - (np.array([gx0, y, z]) + [s, 0, 0])) > thr:
            res = g.round(4); break
    p.chk()
    goto_slow(S, qt=qt, bt=b0, steps=150, vmax=0.06)
    return res

if __name__ == '__main__':
    seed=int(sys.argv[1]); k=None if sys.argv[2]=='n' else int(sys.argv[2]); BX=float(sys.argv[3])
    ys=[float(v) for v in sys.argv[4].split(',')]; zs=[float(v) for v in sys.argv[5].split(',')]
    xs=float(sys.argv[6]) if len(sys.argv)>6 else 1.72; xe=float(sys.argv[7]) if len(sys.argv)>7 else 2.3
    p=P(seed,k); S=p.S; S.grip=1.0
    mode=sys.argv[8] if len(sys.argv)>8 else 'h'
    bi=np.array([xs-0.62-MX-0.3, ys[0], 0]); goto_slow(S, bt=bi, steps=150, vmax=0.1)
    qi=ikh(S,[xs-0.3,ys[0],max(zs[0],0.15)],b=bi); goto_slow(S,qt=qi,bt=bi,steps=300,vmax=0.06)
    print('init gp',grasp_point_world(S).round(3),flush=True)
    for y in ys:
        for z in zs:
            r=bpush(p,y,z,xs,xe,R=None if mode=='h' else Rdown(0.0 if mode=='d' else np.pi/2),reach=0.62 if mode=='h' else 0.5)
            tipx = None if not isinstance(r,np.ndarray) else round(r[0]+TIP,3)
            print('Y %.3f Z %.3f tipx %s raw %s steps %d'%(y,z,tipx,r,len(S.rew)),flush=True)
    print('LOG',LOG)
