import sys, numpy as np, mujoco
exec(open('grasptest.py').read().split("run([ox,0,0.6],8")[0])
run([0,0,0.6],10,float(sys.argv[4]))
sid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
p=d.site_xpos[sid]; R=d.site_xmat[sid].reshape(3,3)
for g in range(m.ngeom):
    nm=m.geom(g).name or m.body(m.geom_bodyid[g]).name
    if not nm.startswith('robot') or (m.geom_contype[g]==0 and m.geom_conaffinity[g]==0): continue
    c=m.geom_aabb[g][:3]; h=m.geom_aabb[g][3:]
    pts=[]
    for sx in (-1,1):
        for sy in (-1,1):
            for sz in (-1,1):
                w=d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@(c+h*np.array([sx,sy,sz]))
                pts.append(R.T@(w-p))
    pts=np.array(pts)
    if pts[:,2].max()>-0.1: print(nm, 'x',pts[:,0].min().round(3),pts[:,0].max().round(3),'y',pts[:,1].min().round(3),pts[:,1].max().round(3),'z(down)',pts[:,2].min().round(3),pts[:,2].max().round(3))
