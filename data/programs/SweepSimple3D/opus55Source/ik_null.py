from ik_tool import *
Rd = np.column_stack([[0,1,0],[1,0,0],[0,0,-1]]).astype(float)
def ikn(p,R,q0,qref,iters=500,w=0.05):
    q=np.array(q0,float)
    for it in range(iters):
        x,Rc=fk(q); ep=p-x
        er=0.5*(np.cross(Rc[:,0],R[:,0])+np.cross(Rc[:,1],R[:,1])+np.cross(Rc[:,2],R[:,2]))
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        Jm=np.vstack([jp[:,armdof],jr[:,armdof]]); e=np.concatenate([ep,er])
        Jp=Jm.T@np.linalg.inv(Jm@Jm.T+1e-4*np.eye(6))
        dq=Jp@e + (np.eye(7)-Jp@Jm)@(w*(qref-q))
        q+=np.clip(dq,-0.1,0.1)
    return q,np.linalg.norm(ep),np.linalg.norm(er)
if __name__=='__main__':
  for r in [0.35,0.4,0.45,0.5,0.55]:
    for z in [0.335,0.37,0.46]:
        q,ep,er=ikn(np.array([r,0,z]),Rd,q_init,q_init)
        print(r,z,'maxdiff %.3f'%np.max(np.abs(q-q_init)), np.round(q-q_init,2), '%.1e %.1e'%(ep,er))
