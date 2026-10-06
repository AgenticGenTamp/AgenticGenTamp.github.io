import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
h=H(32); m,d=h.m,h.d; b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"bowl_0")
V=[]
for g in range(m.ngeom):
    if m.geom_bodyid[g]!=b or m.geom_type[g]!=7 or m.geom_contype[g]==0: continue
    mid=m.geom_dataid[g]; va=m.mesh_vertadr[mid]; vn=m.mesh_vertnum[mid]
    v=m.mesh_vert[va:va+vn]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-d.xpos[b]; V.append(v)
V=np.concatenate(V); r=np.hypot(V[:,0],V[:,1]); z=V[:,2]
print("ncollgeoms",len([1 for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_contype[g]]), "zrange",z.min(),z.max())
for zz in np.arange(-0.019,0.020,0.004):
    s=(z>=zz)&(z<zz+0.004); 
    if s.any(): print(f"z {zz:+.3f}: r min {r[s].min():.3f} max {r[s].max():.3f}")
print("bowl quat", h.obs[3:7])
