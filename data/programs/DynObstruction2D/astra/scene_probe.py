from env_client import make_env
import json
E=make_env()
print('max_steps',E.max_steps)
for seed in range(11):
 s,i=E.reset(seed=seed)
 out={}
 for n in s.get_object_names():
  o=s.get_object_from_name(n)
  fs=['x','y','theta']
  fs+= ['base_radius','arm_joint','arm_length','finger_gap','finger_height','gripper_base_width','gripper_base_height'] if n=='robot' else ['width','height']
  d={}
  for f in fs:
   try: d[f]=round(float(s.get(o,f)),4)
   except Exception: pass
  out[n]=d
 print(seed,json.dumps(out))
E.close()
