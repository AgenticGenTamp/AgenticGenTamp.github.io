from plow_search import *
Rd = np.column_stack([[0,1,0],[1,0,0],[0,0,-1]]).astype(float)
def ikmm(p,R,iters=3000):
    # minimize max |q-q_init| via penalty on largest joint deviation in nullspace
    q=q_init.copy()
    for it in range(iters):
        x,Rc=fk(q); ep=p-x
        er=0.5*(np.cross(Rc[:,0],R[:,0])+np.cross(Rc[:,1],R[:,1])+np.cross(Rc[:,2],R[:,2]))
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        Jm=np.vstack([jp[:,armdof],jr[:,armdof]]); e=np.concatenate([ep,er])
        Jp=Jm.T@np.linalg.inv(Jm@Jm.T+1e-4*np.eye(6))
        dev=q-q_init; w=np.abs(dev)**6; w=w/ (w.max()+1e-12)
        dq=Jp@e + (np.eye(7)-Jp@Jm)@(-0.05*w*np.sign(dev))
        q+=np.clip(dq,-0.05,0.05)
    return q,np.linalg.norm(ep),np.linalg.norm(er)
if __name__=="__main__":
 for r in [0.35,0.45,0.55,0.65,0.75]:
  for yaw in [0]:
    q,ep,er=ikmm(np.array([r,0,0.05]),Rd)
    print(r,'err %.1e %.1e'%(ep,er),'maxdq %.2f'%np.max(np.abs(q-q_init)), np.round(q-q_init,2).tolist(), flush=True)
