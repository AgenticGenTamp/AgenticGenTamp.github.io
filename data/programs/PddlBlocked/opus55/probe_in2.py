import sys; sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open('run_plan.py').read().split("for w in segs:")[0])
for w in segs[:6]:
    follow(w['base'],w['qs'])
    if w['grip']!=0: S.step(np.r_[np.zeros(10),w['grip']])
d=np.r_[g0[:2]-blk[:2],0]; d/=np.linalg.norm(d)
b=S.base()
lat=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
zoff=float(sys.argv[3]) if len(sys.argv)>3 else 0.0
perp=np.array([-d[1],d[0],0])
start=np.r_[blk[:2],g0[2]]-0.10*d
from planner import cart_path
qs=cart_path(b,S.q(),fk_world(b,S.q())[0],start+lat*perp+[0,0,zoff],d); follow(b,qs)
for k in range(1,60):
    p0=fk_world(b,S.q())[0]
    qs=cart_path(b,S.q(),p0,p0+0.005*d,d)
    if not follow(b,qs): break
p,R=fk_world(b,S.q())
print('lat',lat,'zoff',zoff,'advanced',round((p-start)@d,3),'tool-to-g0 along d',round((g0[:3]-p)@d,3))
print('pts',[x.round(3) for x in fk_points(b,S.q())], 'base',b.round(3))
S.step(np.r_[np.zeros(10),-1.0]); print('grasp',S.rget('grasp_active'))
