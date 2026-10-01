from env_client import make_env
import numpy as np

def vals(o,n,features):
 b=o.get_object_from_name(n); return [round(float(o.get(b,f)),5) for f in features]
def dump(o,env):
 for name in o.get_object_names():
  ob=o.get_object_from_name(name)
  print(name,repr(ob))
  for ty in env.observation_space.types:
   try:
    if ob in o.get_objects(ty): print([(f,round(float(o.get(ob,f)),5)) for f in env.observation_space.type_features[ty]])
   except Exception as e: print(type(e).__name__,str(e))
env=make_env(); o,info=env.reset(seed=0); print('INFO',info); dump(o,env)
f=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]+['grasp_tf_x','grasp_tf_y','grasp_tf_z','grasp_tf_qx','grasp_tf_qy','grasp_tf_qz','grasp_tf_qw']
for k in range(10):
 o,info=env.reset(seed=0)
 for step in range(3):
  a=np.zeros(11);a[k]=.2;o,r,t,tr,info=env.step(a); print('ACTION',k,step,vals(o,'robot',f), info)
env.close()
