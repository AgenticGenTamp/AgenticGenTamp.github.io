import sys; sys.path.insert(0,'.')
from env_client import make_env
from ctrl import *
roll=float(sys.argv[1]); grip=float(sys.argv[2]); z=float(sys.argv[3])
env = make_env(); obs, info = env.reset(seed=11, options={'object_count':1}); S = Sim(env, obs)
cy=0.0
a=np.radians(roll); zz=np.array([1.,0,0]); xx=np.array([0,np.cos(a),np.sin(a)]); R=np.column_stack([xx,np.cross(zz,xx),zz])
S.goto(grip=grip, steps=3)
e,_=move_ee_world(S, np.array([1.55,cy,z]), R, bt=np.array([1.55-MX-0.62,cy,0]), steps=250)
print('ok',e, np.round(grasp_point_world(S),3))
q0=S.q().copy(); b0=S.base().copy()
for k in range(1,50):
    S.goto(qt=q0, bt=b0+np.array([0.01*k,0,0]), steps=6, tol=0.003)
    if S.base()[0] < b0[0]+0.01*k-0.015: break
print('roll',roll,'grip',grip,'blocked gp', np.round(grasp_point_world(S),3))
