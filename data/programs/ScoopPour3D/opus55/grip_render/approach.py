import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kin
RD = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
JN = ['pos_arm_joint%d'%i for i in range(1,8)]
CUBE = os.environ.get('GCUBE','cube_14'); DZ=float(os.environ.get('GDZ','0.0')); OPEN=float(os.environ.get('GOPEN','0.0'))
def cyaw(q):
    w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
class GeneratedApproach:
    def __init__(s, *a, **k): pass
    def reset(s, obs, info):
        s.obs=obs; s.gen=s.script(); s.I=np.zeros(7); s.qi=s.q(); s.qs=s.qi.copy(); s.g=0.0; s.qprev=s.qi
    def q(s): r=s.obs.get_object_from_name('robot'); return np.array([s.obs.get(r,j) for j in JN])
    def base(s): r=s.obs.get_object_from_name('robot'); return np.array([s.obs.get(r,k) for k in ('pos_base_x','pos_base_y','pos_base_rot')])
    def P(s,n): o=s.obs.get_object_from_name(n); return np.array([s.obs.get(o,f) for f in 'xyz'])
    def Q(s,n): o=s.obs.get_object_from_name(n); return np.array([s.obs.get(o,f) for f in ('qw','qx','qy','qz')])
    def act(s, db, dq):
        a=np.zeros(11,dtype=np.float32); a[0:3]=np.clip(db,-.1,.1); a[3:10]=np.clip(dq,-.1,.1); s.qi=s.qi+0.25*a[3:10]; a[10]=s.g; return a
    def ctrl(s, pw, Rw, bt, tol, maxsteps):
        for k in range(maxsteps):
            b=s.base(); d=bt-b
            qt,_=kin.ik_arm(kin.world_to_arm(pw,bt),Rw,s.qs,iters=100,lam=0.05); s.qs=qt
            q=s.q(); qp=s.qprev; s.qprev=q
            if np.max(np.abs(qt+s.I-s.qi))<0.05 and np.max(np.abs(q-qp))<0.003: s.I=np.clip(s.I+0.3*(qt-q),-.2,.2)
            qc=qt+s.I
            yield s.act([d[0]/.87,d[1]/.87,d[2]], (qc-s.qi)/0.25)
            p,_=kin.fk_world(s.base(),s.q())
            if np.max(np.abs(qc-s.qi))<1e-3 and np.linalg.norm(p-pw)<tol: return
    def hold(s,n):
        for k in range(n): yield s.act([0,0,0],(s.qs+s.I-s.qi)/0.25)
    def script(s):
        c0=s.P(CUBE); y=(cyaw(s.Q(CUBE))+np.pi/4)%(np.pi/2)-np.pi/4; Rw=kin.rotz(y)@RD; bt=np.array([-0.15,c0[1],0])
        s.g=OPEN
        yield from s.ctrl(c0+[0,0,0.56-c0[2]],Rw,bt,0.01,120)
        yield from s.ctrl(c0+[0,0,0.03],Rw,bt,0.004,60)
        yield from s.ctrl(c0+[0,0,DZ],Rw,bt,0.002,60)
        yield from s.hold(3)
        s.g=1.0
        yield from s.hold(10)
        yield from s.ctrl(c0+[0,0,0.56-c0[2]],Rw,bt,0.02,60)
        while True: yield from s.hold(1)
    def get_action(s, obs):
        s.obs=obs; return next(s.gen)
