from test_batch import *
import numpy as np
from kinova_candidate import fk

def run_cal(args):
 seed,count=args;e=make_env();s,i=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);shots=[];shot=None
 try:
  for k in range(e.max_steps):
   a=p.get_action(s)
   if p.phase=='kick':
    q=np.array([s.get(p.robot,'pos_arm_joint'+str(j)) for j in range(1,8)])
    base=np.array([s.get(p.robot,f) for f in ['pos_base_x','pos_base_y', 'pos_base_rot']])
    pos=p.xyz(s,p.cube);pred=fk(q,.12,(.1,0,.35))[0]+[base[0],base[1],0]
    shot={'initial':pos.tolist(),'off':(pos-pred).tolist(),'base':base.tolist(),'cube':p.cube.name}
   s,r,t,tr,i=e.step(a)
   if p.phase=='flight' and p.t==1 and shot is not None:
    pos=p.xyz(s,p.cube);v=np.array([s.get(p.cube,f) for f in ['vx','vy','vz']]);tf=(v[2]+np.sqrt(max(0,v[2]**2+19.62*(pos[2]-.05))))/9.81
    land=pos[:2]+v[:2]*tf
    shot.update({'release':pos.tolist(),'v':v.tolist(),'dx':float(land[0]-shot['initial'][0]),'land':land.tolist(),'bin':p.xyz(s,p.bin).tolist()});shots.append(shot);shot=None
   if t or tr or (p.phase=='select' and all(c.name in p.done for c in p.cubes)):break
  return {'seed':seed,'count':count,'success':t,'steps':k+1,'shots':shots}
 finally:e.close()
if __name__=='__main__':
 start=int(sys.argv[1]);n=int(sys.argv[2])
 with ThreadPoolExecutor(max_workers=4) as pool:
  for out in pool.map(run_cal,[(s,c) for s in range(start,start+n) for c in [1,2]]):print(json.dumps(out),flush=True)
