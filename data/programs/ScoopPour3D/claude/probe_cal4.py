from env_client import make_env
import numpy as np, kin
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); SC = obs.get_object_from_name('scoop_0')
base = obs.data[R][:3].copy(); th=base[2]
scoop0 = obs.data[SC][:3].copy()
dx,dy = scoop0[0]-base[0], scoop0[1]-base[1]
fx = np.cos(th)*dx+np.sin(th)*dy; fy=-np.sin(th)*dx+np.cos(th)*dy
print('base',np.round(base,3),'scoop world',np.round(scoop0,3),'scoop base-frame',round(fx,3),round(fy,3))
TOOL=0.12
q = obs.data[R][3:10].copy()
def goto(tgt, n=60, tol=0.008):
    global obs,q
    qd,err = kin.ik(np.array(tgt), kin.rot_down(), q, TOOL)
    for k in range(n):
        qc = obs.data[R][3:10]
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-qc,-0.1,0.1); a[10]=0.0
        obs,_,_,_,_=env.step(a)
        if np.abs(qd-obs.data[R][3:10]).max()<tol: break
    q = obs.data[R][3:10].copy()
    return err, np.abs(qd-q).max()
goto((fx, fy-0.18, 0.30))
goto((fx, fy-0.18, 0.20))
print('descended, scoop', np.round(obs.data[SC][:3]-scoop0,3))
for ty in np.arange(fy-0.16, fy+0.20, 0.02):
    e,qe = goto((fx,ty,0.20),25)
    print('ty',round(ty,3),'qerr',round(qe,3),'scoop d',np.round(obs.data[SC][:3]-scoop0,3))
env.close()
