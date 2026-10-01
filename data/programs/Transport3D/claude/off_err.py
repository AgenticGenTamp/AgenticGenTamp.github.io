import numpy as np, kutil, kin, off_lib as L
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
def quat2R(q):  # xyzw
    return kin.quat_to_mat(np.asarray(q,float),wxyz=False)
env=make_env(); o,info=env.reset(seed=0)
c=kutil.Ctl(env,o)
def cube_pose():
    ob=c.o.get_object_from_name("cube1")
    p=np.array([float(c.o.get(ob,"pose_"+f)) for f in "xyz"])
    qt=np.array([float(c.o.get(ob,"pose_q"+f)) for f in ["x","y","z","w"]])
    return p,qt
def gtf():
    R=c.o.get_object_from_name("robot")
    p=np.array([float(c.o.get(R,"grasp_tf_"+f)) for f in "xyz"])
    qt=np.array([float(c.o.get(R,"grasp_tf_q"+f)) for f in ["x","y","z","w"]])
    return p,qt
cu=c.opos("cube1")
c.gotobase(-0.15,-0.24,3.13); b,q,g=c.robot(); yaw=b[2]
L.safe_to(c,cu[0],cu[1],cu[2],yaw); c.grip(True)
gp,gq=gtf(); print("gtf",np.round(gp,5),np.round(gq,5))
rows=[]
for (dx,dy,dz,yw) in [(0,0,0.15,yaw),(0.05,0.05,0.25,yaw),(-0.05,0.08,0.20,yaw+0.6),
                      (0.10,-0.05,0.30,yaw-0.6),(0.0,0.0,0.35,yaw+1.5),(0.08,0.08,0.18,yaw-1.5)]:
    ok=L.cmove(c,[cu[0]+dx,cu[1]+dy,cu[2]+dz],yw)
    T,qq,bb=L.fkpos(c)
    cp,cq=cube_pose()
    # predicted cube pos assuming gtf = cube in tool frame
    pred=T[:3,3]+T[:3,:3]@gp
    rows.append((ok,T[:3,3].copy(),cp.copy(),pred.copy(),qq.copy(),T[:3,:3].copy(),cq.copy()))
    print("ok",ok,"fk",np.round(T[:3,3],4),"cube",np.round(cp,4),"pred",np.round(pred,4),
          "resid_world",np.round(cp-pred,4))
np.save("off_err.npy",np.array([np.concatenate([r[1],r[2],r[3],r[4],r[5].ravel(),r[6]]) for r in rows]))
print("steps",c.steps)
env.close()
