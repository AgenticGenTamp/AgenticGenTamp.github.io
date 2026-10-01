from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=0); rob=s.get_object_from_name('robot')
def vals():return [round(s.get(rob,k),5) for k in ['pos_base_x','pos_base_y','joint_2','joint_4','finger_state','grasp_active']]
def act(x):
 global s
 s,r,te,tr,inf=E.step(np.array(x));return te
for v in [1,-1,1]:
 a=np.zeros(11);a[10]=v;act(a);print(v,vals())
# target part0 y .289, initial x .313: scan x/y joint2 and4
for by in [.289,.339,.239,-.33,-.28,-.38,0]:
 for j2 in [-.35,-.15,-.55,-.75,.05]:
  for j4 in [-2.5,-2.3,-2.7,-2.1]:
   for _ in range(5):
    a=np.zeros(11);a[1]=np.clip(by-s.get(rob,'pos_base_y'),-.2,.2);a[4]=np.clip(j2-s.get(rob,'joint_2'),-.2,.2);a[6]=np.clip(j4-s.get(rob,'joint_4'),-.2,.2);a[10]=1;act(a)
   a=np.zeros(11);a[10]=-1;act(a)
   if s.get(rob,'grasp_active'):
    print('GRASP',vals());
    for name in s.get_object_names():
     o=s.get_object_from_name(name)
     if name.startswith('part') or name=='rack':print(name,[s.get(o,k) for k in ['pose_x','pose_y','pose_z','grasp_active']])
    E.close();quit()
print('NONE');E.close()
