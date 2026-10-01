from explore_cup_lib import *
from kin import jac_pos_rot, rot_err
LIM=np.array([9,2.31,9,2.56,9,2.13,9])
def Rp(p):  # approach tilted down by p from +x
    z=np.array([np.cos(p),0,-np.sin(p)]); x=np.array([0,1.,0]); y=np.cross(z,x); return np.stack([x,y,z],1)
def ikl(pb,R,q0,iters=300):
    f=pb-np.array([MX,0,H])-L*R[:,2]; q=np.array(q0,float)
    for _ in range(iters):
        T,J=jac_pos_rot(q); ep=f-T[:3,3]; er=rot_err(T[:3,:3],R)
        e=np.concatenate([ep,0.5*er]); Jw=J.copy(); Jw[3:]*=0.5
        dq=Jw.T@np.linalg.solve(Jw@Jw.T+1e-3*np.eye(6),e)
        n=np.linalg.norm(dq); dq*=min(1,0.2/n) if n>0 else 1
        q=np.clip(q+dq,-LIM,LIM)
    return q,np.linalg.norm(ep),np.linalg.norm(er)
if __name__=='__main__':
    rng=np.random.default_rng(0)
    np.set_printoptions(precision=2,suppress=True)
    for pdeg in [0,30,45]:
      for z in [0.08,0.35,0.62]:
        for x in [0.55,0.7,0.85]:
            best=None
            for k in range(8):
                q0=np.array([0,0.3,3.14,-2.0,0,1.0,1.57])+rng.normal(0,0.6,7)*(k>0)
                q,e1,e2=ikl(np.array([x,0,z]),Rp(np.radians(pdeg)),q0)
                if best is None or e1+e2<best[1]+best[2]: best=(q,e1,e2)
            print(pdeg,z,x,round(best[1],3),round(best[2],3),best[0])
