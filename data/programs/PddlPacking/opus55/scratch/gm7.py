# holding-block clearance: seed 9, grasp block A, sweep held block toward block B
import sys; sys.path.insert(0,'scratch')
from gm_lib import *
def blocks(o):
    out=[]
    for b in o.get_objects(T("block")):
        g=lambda k: float(o.get(b,k)); out.append((np.array([g("pose_x"),g("pose_y"),g("pose_z")]),2*math.atan2(g("pose_qz"),g("pose_qw")),g("grasp_active")))
    return out
zrel=float(sys.argv[1]); dname=sys.argv[2]; tz=float(sys.argv[3]) if len(sys.argv)>3 else 0.80
r=R(9,(-0.47,-0.15,0.0))
(A,ay,_),(B,byw,_)=blocks(r.o)[:2]
assert r.move_tool([A[0],A[1],0.95],ay)
assert r.move_tool([A[0],A[1],tz],ay)
r.grip(-1.0); rs=rstate(r.o)
held=[b[2] for b in blocks(r.o)]
assert r.move_tool([A[0],A[1],0.95],ay)
yaw=byw; c,s=math.cos(yaw),math.sin(yaw)
Y=np.array([c,s,0.]); Z=np.array([s,-c,0.])
u={"+y":Y,"-y":-Y,"+z":Z,"-z":-Z}[dname]
zt=B[2]+zrel   # tool z
pn=np.array([B[0],B[1],zt]); pf=pn+0.16*u
d=r.sweep(pf,pn,yaw)
hb=[b for b in blocks(r.o) if b[2]>0.5]
p,Rm=fk(r.base,r.cfg())
print("grasp tz %.3f opening %.4f grasp_tf %s | zt %.4f dir %s clearance %.4f | held centre-tool z %.4f"%(tz,rs[10],rs[12:15].round(4),zt,dname,d,(hb[0][0][2]-p[2]) if hb else float('nan')),flush=True)
