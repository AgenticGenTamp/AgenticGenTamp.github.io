from ik_null import *
import itertools
geoms=[g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or '').startswith('robot') and m.geom_type[g]==7 and m.geom_contype[g]]
def verts(q):
    fk(q)
    out=[]
    for g in geoms:
        mid=m.geom_dataid[g]; a=m.mesh_vertadr[mid]; n=m.mesh_vertnum[mid]
        V=m.mesh_vert[a:a+n]
        out.append(V@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    # pads (boxes) too
    return np.vstack(out)
def profile(q, zc=0.025):
    V=verts(q)
    low=V[V[:,2]<zc]
    if len(low)==0: return None
    return V[:,2].min(), low[:,0].min(), low[:,0].max(), low[:,1].min(), low[:,1].max()
def rot(app, close):
    z=np.array(app,float); x=np.array(close,float); y=np.cross(z,x)
    return np.column_stack([x,y,z])
if __name__=='__main__':
    res=[]
    for app in [(0,1,0),(0,-1,0)]:
      for close in [(0,0,1),(1,0,0)]:
        R=rot(app,close)
        for xp in [0.4,0.5,0.6]:
          for yp in [-0.2,-0.1,0.0,0.1,0.2]:
            for zp in [0.03,0.045,0.06]:
                q,ep,er=ikn(np.array([xp,yp,zp]),R,q_init,q_init,iters=400)
                if ep>2e-3 or er>2e-3: continue
                pr=profile(q)
                if pr is None: continue
                zmin,x0,x1,y0,y1=pr
                print(app,close,xp,yp,zp,'zmin %.3f low x[%.2f,%.2f] y[%.2f,%.2f] width %.2f maxdq %.2f'%(zmin,x0,x1,y0,y1,y1-y0,np.max(np.abs(q-q_init))), flush=True)
