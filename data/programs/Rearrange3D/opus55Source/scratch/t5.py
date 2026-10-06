import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
h=H(int(sys.argv[1])); h.step(60); m,d=h.m,h.d
def body(name): return mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
def waabb(bid):
    lo=np.full(3,9.);hi=np.full(3,-9.)
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=bid: continue
        if m.geom_type[g]==7:
            mid=m.geom_dataid[g]; va=m.mesh_vertadr[mid]; vn=m.mesh_vertnum[mid]
            v=m.mesh_vert[va:va+vn]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        else:
            v=d.geom_xpos[g][None]
        lo=np.minimum(lo,v.min(0)); hi=np.maximum(hi,v.max(0))
    return np.round(lo,3),np.round(hi,3)
for n in ["boxed_drink_0","bowl_0","can_0"]: print(n, waabb(body(n)), "conaffinity", [m.geom_contype[g] for g in range(m.ngeom) if m.geom_bodyid[g]==body(n)][:3], "fric",[m.geom_friction[g][0] for g in range(m.ngeom) if m.geom_bodyid[g]==body(n)][:2])
# contacts of drink
dr=body("boxed_drink_0")
for i in range(d.ncon):
    c=d.contact[i]; b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
    if dr in (b1,b2):
        o=b2 if b1==dr else b1; print("drink contact with", mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,o), mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1 if b1!=dr else c.geom2), np.round(c.pos,3), round(c.dist,4))
