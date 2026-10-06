import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
h=H(32); h.step(60); m,d=h.m,h.d; b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"boxed_drink_0")
V=[]
for g in range(m.ngeom):
    if m.geom_bodyid[g]!=b or m.geom_type[g]!=7 or m.geom_contype[g]==0: continue
    mid=m.geom_dataid[g]; va=m.mesh_vertadr[mid]; vn=m.mesh_vertnum[mid]
    V.append(m.mesh_vert[va:va+vn]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
V=np.concatenate(V)
for zz in np.arange(0.47,0.512,0.004):
    s=(V[:,2]>=zz)&(V[:,2]<zz+0.004)
    if s.any(): print(f"z {zz:.3f}: x [{V[s,0].min():.3f},{V[s,0].max():.3f}] y [{V[s,1].min():.3f},{V[s,1].max():.3f}] n {s.sum()}")
