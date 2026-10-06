import numpy as np, sys
sys.path.insert(0,'.')
from scipy.optimize import minimize
from approach import R_hook
from kin import make_model, Kin, Q_RETRACT, LIM, rot_err
env, model = make_model()
kin = Kin(model)
q0 = Q_RETRACT
bx=float(sys.argv[1]) if len(sys.argv)>1 else 1.42
def pose_res(x):
    q=x[:7]; a=x[7]; by=x[8]
    R=R_hook(a); dv=R[:,2]
    site=np.array([0.912,0,0.322])-0.02*dv+[0,0,0.08]
    p,Rc=kin.fk(np.array([bx,by,np.pi]),q)
    return np.concatenate([p-site, rot_err(Rc,R)])
def obj(x):
    return x[9]
cons=[{'type':'eq','fun':pose_res}]
for j in range(7):
    cons.append({'type':'ineq','fun':lambda x,j=j: x[9]-(x[j]-q0[j])})
    cons.append({'type':'ineq','fun':lambda x,j=j: x[9]+(x[j]-q0[j])})
bnds=[tuple(LIM[j]) if LIM[j,0]>-1e8 else (None,None) for j in range(7)]+[(0.5,0.5),(-0.33,-0.33),(0,3)]
best=None
starts=[[1.266,0.886,2.671,-1.984,-0.889,-0.075,0.315],
        list(q0+np.array([0.89,1.16,0.07,0.55,-1.54,0.43,-0.74]))]
rng=np.random.default_rng(1)
for k in range(10):
    s = starts[k] if k<2 else list(q0+rng.uniform(-1,1,7))
    x0=np.array(s+[0.5,-0.33,1.6])
    r=minimize(obj,x0,constraints=cons,bounds=bnds,method='SLSQP',options={'maxiter':300})
    if r.success and np.linalg.norm(pose_res(r.x))<1e-3:
        if best is None or r.x[9]<best[9]: best=r.x
        print(k, round(r.x[9],3), np.round(r.x[7:9],3))
print("best", list(np.round(best,4)))
