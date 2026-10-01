"""Rod-tip horizontal probing with pitch compensation. usage: seed k(n) BX 'y1,y2' 'z1,z2,..'"""
import sys; sys.path.insert(0,'/sandbox/scratch')
from plib import *

def rod_axis(r):
    w, x, y, z = r[3:7]
    return np.array([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)])

def rotmin(a, b):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b); v = np.cross(a, b); c = a@b
    V = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + V + V@V/(1+c)

class RP:
    def __init__(self, p, name):
        self.p = p; self.S = p.S; self.name = name; self.R = Rdown(np.pi/2)
    def tipdir(self):
        a = rod_axis(self.S.rods()[self.name]); return a if a[0] > 0 else -a
    def level(self):
        # correct R so that rod axis -> +x
        for _ in range(3):
            a = self.tipdir()
            if np.linalg.norm(a - [1, 0, 0]) < 0.005: break; self.R = rotmin(a, np.array([1., 0, 0])) @ self.R
            g = grasp_point_world(self.S); e1, n = moveto_slow(self.p, g, self.R, steps=60, vmax=0.03)
            from kin import fk
            Rc = R_world_to_arm(self.S.base(), np.eye(3)).T @ fk(self.S.q(), TOOL)[:3,:3]
            print('    level e1 %.4f n %d  Rerr %.3f dir %s' % (e1, n, np.linalg.norm(Rc - self.R), self.tipdir().round(3)))
    def tip(self):
        r = self.S.rods()[self.name]; return r[:3] + 0.15*self.tipdir()
    def goto_c(self, c, steps=250, vmax=0.05, bt=None):
        """move rod center (via grasp point offset) to c"""
        off = self.S.rods()[self.name][:3] - grasp_point_world(self.S)
        return moveto_slow(self.p, np.asarray(c) - off, self.R, bt=bt, steps=steps, vmax=vmax)
    def push(self, y, z, xc0, xc1, v=0.001, ang=0.008, lagt=0.006):
        S = self.S
        self.goto_c([xc0, y, z])
        a0 = self.tipdir(); b = S.base()
        g0 = grasp_point_world(S); x = g0[0]; t0 = self.tip()
        while x < xc1 - (self.S.rods()[self.name][0] - g0[0]):
            x += v; tgt = np.array([x, g0[1], g0[2]])
            qt, _, _ = ik(S.q(), world_to_arm(b, tgt), R_world_to_arm(b, self.R), tool=TOOL)
            e = qt - S.q(); e[[0,2,4,6]] = (e[[0,2,4,6]] + np.pi) % (2*np.pi) - np.pi
            a = np.zeros(11); a[10] = S.grip; a[3:10] = np.clip(e, -0.1, 0.1); S.step(a)
            tp = self.tip(); da = np.linalg.norm(self.tipdir() - a0); gp = grasp_point_world(S)
            slide = abs((tp[0] - gp[0]) - (t0[0] - g0[0]))
            lag = max(x - gp[0] - 0.006, slide * 3)
            if da > ang or lag > lagt:
                self.p.chk(); res = tp.round(3)
                print('   at contact dir', self.tipdir().round(3))
                self.goto_c([xc0, y, z], steps=120, vmax=0.03); print('   after back', self.tipdir().round(3), self.tip().round(3)); self.level(); print('   after level', self.tipdir().round(3))
                return res, round(da, 3), round(lag, 3)
        self.p.chk(); self.goto_c([xc0, y, z], steps=120, vmax=0.03)
        return None

if __name__ == '__main__':
    seed=int(sys.argv[1]); k=None if sys.argv[2]=='n' else int(sys.argv[2]); BX=float(sys.argv[3])
    ys=[float(v) for v in sys.argv[4].split(',')]; zs=[float(v) for v in sys.argv[5].split(',')]
    p=P(seed,k); S=p.S
    rods=sorted(S.rods(), key=lambda n: np.linalg.norm(S.rods()[n][:2]-S.base()[:2]))
    p.pick(rods[0]); rp=RP(p,rods[0])
    g=grasp_point_world(S); moveto_slow(p,[g[0],g[1],0.4],rp.R); rp.level()
    for y in ys:
        rp.goto_c([BX+0.45,y,0.35],bt=[BX,y,0]); rp.level()
        for z in zs:
            r=rp.push(y,z,BX+0.45,BX+0.9); print('  dir',rp.tipdir().round(3),'tip',rp.tip().round(3))
            print('Y %.3f Z %.3f -> %s  held %s steps %d'%(y,z,r,np.round(S.rods()[rods[0]][:3]-grasp_point_world(S),3),len(S.rew)),flush=True)
    print('LOG',LOG)
