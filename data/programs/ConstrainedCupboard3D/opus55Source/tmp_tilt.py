import numpy as np, mujoco
import approach as ap
K=ap._Kin.get(); m,d=K.m,K.d
def Rtilt(a):
    # site axes: x=finger axis=world y; z=approach=(sin a,0,-cos a); y = z cross x
    x=np.array([0,1.,0]); z=np.array([np.sin(a),0,-np.cos(a)]); y=np.cross(z,x)
    return np.column_stack([x,y,z])
seed=ap.HOME_Q
for a in [0, 0.4, 0.6, 0.8]:
  for reach in [0.5,0.57,0.65]:
    for zz in [0.03,0.075,0.2,0.3]:
        q,err=K.ik(np.array([reach,0,zz]),Rtilt(a),seed)
        p,R=K.fk(q)
        mx=-9; who=''
        for g in range(m.ngeom):
            if m.geom_contype[g]==0 and m.geom_conaffinity[g]==0: continue
            b=m.body(m.geom_bodyid[g]).name
            if b in ('base_link','world'): continue
            Rg=d.geom_xmat[g].reshape(3,3)
            # use rbound sphere approx: center +- rbound
            # better: use aabb
            c=d.geom_xpos[g]+Rg@m.geom_aabb[g][:3]; ext=np.abs(Rg)@m.geom_aabb[g][3:]
            if c[2]-ext[2] < 0.58:
                if c[0]+ext[0]-p[0] > mx: mx=c[0]+ext[0]-p[0]; who=b
        print(f"a={a} reach={reach} z={zz} err={err[0]:.4f},{err[1]:.3f} maxx_rel={mx:.3f} ({who}) q={np.round(q,2)}")
print('----')
for a in [0.5,0.6,0.7]:
    q,err=K.ik(np.array([0.65,0,0.1]),Rtilt(a),seed); p,R=K.fk(q)
    for g in range(m.ngeom):
        b=m.body(m.geom_bodyid[g]).name
        if 'pad' in b or 'follower' in b:
            Rg=d.geom_xmat[g].reshape(3,3); c=d.geom_xpos[g]+Rg@m.geom_aabb[g][:3]; ext=np.abs(Rg)@m.geom_aabb[g][3:]
            print(a, m.geom(g).name or b, 'zmin rel', round(c[2]-ext[2]-p[2],3), 'x range rel', round(c[0]-ext[0]-p[0],3), round(c[0]+ext[0]-p[0],3))
