from env_client import make_env
import numpy as np,json
E=make_env();s,_=E.reset(seed=0);rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0')
features=['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]
target=np.array([-.22,.287928909,0,0,.391676098,-np.pi,-2.05185294,0,-.452491313,np.pi/2]);data=[]
def act(a):
 global s
 s,*_=E.step(a)
def record(tag):
 data.append({'tag':tag,'robot':{k:float(s.get(rob,k)) for k in E.observation_space.type_features[rob.type]},'part':{k:float(s.get(p,k)) for k in E.observation_space.type_features[p.type]}})
for k in range(12):
 a=np.zeros(11);a[:10]=np.clip(target-np.array([s.get(rob,k) for k in features]),-.2,.2);act(a)
a=np.zeros(11);a[10]=-1;act(a);record('baseline')
a=np.zeros(11);a[4]=-.15;a[8]=-.15;act(a);record('lift')
for j in range(10):
 for sign in [-1,1]:
  a=np.zeros(11);a[j]=sign*.05;act(a);record(str(j)+'_'+str(sign));a[j]=-a[j];act(a)
with open('calibration_data.json','w') as f:json.dump(data,f)
print('wrote',len(data),'grasp',s.get(rob,'grasp_active'));E.close()
