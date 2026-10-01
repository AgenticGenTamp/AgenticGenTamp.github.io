import numpy as np, fk
q0 = np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
for lift in [0.0,0.1,0.2,0.3,0.33]:
    p,R,_ = fk.fk_world((-1.,0.,0.), q0, lift)
    print(lift, np.round(p,4))
    print(np.round(R,3))
    break
# check jacobian numerically
lift=0.3
J = fk.jacobian(q0, lift)
Jn = np.zeros((3,7))
p0,_,_ = fk.fk_arm(q0,lift)
for i in range(7):
    qq=q0.copy(); qq[i]+=1e-6
    p1,_,_=fk.fk_arm(qq,lift)
    Jn[:,i]=(p1-p0)/1e-6
print("jac err", np.abs(J[:3]-Jn).max())
