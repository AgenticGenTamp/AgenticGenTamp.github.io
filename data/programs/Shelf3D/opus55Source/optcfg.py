import numpy as np, mujoco
from scipy.optimize import minimize
m=mujoco.MjModel.from_xml_path("/robocode/third-party/kindergarden/src/kinder/envs/dynamic3d/models/stanford_tidybot/tidybot.xml")
d=mujoco.MjData(m)
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"pinch_site")
q0=np.array([0,-0.349,3.142,-2.548,0,-0.873,1.571])
# geoms to check in shelf: all arm geoms with collision
arm_geoms=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g]>0 and m.body(m.geom_bodyid[g]).name not in ('base_link',)]
def verts(g):
    mid=m.geom_dataid[g]
    if m.geom_type[g]==7:
        va=m.mesh_vertadr[mid]; vn=m.mesh_vertnum[mid]; v=m.mesh_vert[va:va+vn]
    else:
        s=m.geom_size[g]; v=np.array([[a,b,c] for a in (-s[0],s[0]) for b in (-s[1],s[1]) for c in (-s[2],s[2])])
    R=d.geom_xmat[g].reshape(3,3); return v@R.T+d.geom_xpos[g]
def fk(q):
    d.qpos[:]=0; d.qpos[3:10]=q; mujoco.mj_kinematics(m,d)
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3,3).copy()
CUBE_OFF=0.015
def cube_center(q):
    p,R=fk(q); return p+CUBE_OFF*R[:,2], R
ZPICK=0.02; ZPLACE=0.578
def cons(x, phi_extra):
    qa,qb=x[:7],x[7:14]
    ca,Ra=cube_center(qa); cb,Rb=cube_center(qb)
    th=x[14]; ph=th+phi_extra
    out=[ca[1], ca[2]-ZPICK, cb[1], cb[2]-ZPLACE]
    out+=list(Ra[:,0]-np.array([0,np.sign(Ra[1,0]) or 1,0]))[0:1]+[Ra[2,0]]
    out+=list(Ra[:,2]-np.array([np.sin(th),0,-np.cos(th)]))
    out+=[Rb[0,0],Rb[2,0]]
    out+=list(Rb[:,2]-np.array([np.sin(ph),0,-np.cos(ph)]))
    # same finger axis sign
    out+=[Ra[1,0]-Rb[1,0]]
    return np.array(out)
def shelf_clear(x):
    qb=x[7:14]; cb,Rb=cube_center(qb)
    front=cb[0]-(1.42-1.373)-0.005
    pen=[]
    for g in arm_geoms:
        v=verts(g); xm=v[:,0].max()
        pen.append(max(front-xm, 0.795-v[:,2].max())); pen.append(max(front-xm, v[:,2].min()-0.553))
    return np.array(pen)
def pick_clear(x):
    qa=x[:7]; cube_center(qa)
    return np.array([verts(g)[:,2].min()-0.004 for g in arm_geoms])
lim=np.array([[-1e9,1e9],[-2.2,2.2],[-1e9,1e9],[-2.5,2.5],[-1e9,1e9],[-2.05,2.05],[-1e9,1e9]])
best=None
rng=np.random.default_rng(0)
for phi_extra in [0.0, np.pi/2]:
  for trial in range(12):
    th0=rng.uniform(-0.2,0.8)
    x0=np.concatenate([q0+rng.normal(0,0.15,7), q0+rng.normal(0,0.15,7),[th0, 0.5]])
    obj=lambda x: x[15]
    cl=[{'type':'eq','fun':lambda x,p=phi_extra: cons(x,p)},
        {'type':'ineq','fun':lambda x: x[15]-np.abs(x[:7]-x[7:14])},
        {'type':'ineq','fun':shelf_clear},
        {'type':'ineq','fun':pick_clear},
        {'type':'ineq','fun':lambda x: cube_center(x[7:14])[0][0]-0.36},
        {'type':'ineq','fun':lambda x: cube_center(x[:7])[0][0]-0.36},
        {'type':'ineq','fun':lambda x: np.concatenate([x[:14]-np.tile(lim[:,0],2), np.tile(lim[:,1],2)-x[:14]])}]
    r=minimize(obj,x0,constraints=cl,method='SLSQP',options={'maxiter':400})
    ok = r.success and np.abs(cons(r.x,phi_extra)).max()<1e-4 and shelf_clear(r.x).min()>-1e-4 and pick_clear(r.x).min()>-1e-4
    print(phi_extra, trial, ok, round(r.x[15],3), 'th',round(r.x[14],2), 'ra',cube_center(r.x[:7])[0].round(3),'rb',cube_center(r.x[7:14])[0].round(3))
    if ok and (best is None or r.x[15]<best[0][15]): best=(r.x,phi_extra)
x,p=best
print("BEST", x[15], p); print("qpick",list(np.round(x[:7],4))); print("qplace",list(np.round(x[7:14],4)))
print("ra",cube_center(x[:7])[0], "rb",cube_center(x[7:14])[0])
np.save("best_cfg.npy", np.concatenate([x,[p]]))
