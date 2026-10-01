import numpy as np
from env_client import make_env

def vals(s):
    r=s.get_object_from_name('robot')
    gf=lambda f: float(s.get(r,f))
    return np.array([gf('pos_base_x'),gf('pos_base_y'),gf('pos_base_rot')]+[gf('pos_arm_joint'+str(i)) for i in range(1,8)]+[gf('pos_gripper')])

def run(action, n=10, seed=1):
    e=make_env(); s,_=e.reset(seed=seed); v0=vals(s); print('start',np.round(v0,3))
    for i in range(n):
        s,r,t,tr,info=e.step(np.array(action,dtype=np.float32))
        if i in (0,1,4,9): print(i+1,'v',np.round(vals(s),3),'r',r)
    e.close()

if __name__=='__main__':
  for j in [0,1,2,3,9,10]:
    print('\nA',j); a=np.zeros(11); a[j]=1 if j==10 else .1; run(a)
