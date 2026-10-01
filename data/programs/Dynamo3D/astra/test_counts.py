from env_client import make_env
from approach import GeneratedApproach
import time, json, sys
count=int(sys.argv[1]); start_seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
num=int(sys.argv[3]) if len(sys.argv)>3 else 20
with make_env() as env:
 p=GeneratedApproach(env.action_space,env.observation_space,{})
 for seed in range(start_seed,start_seed+num):
  s,info=env.reset(seed=seed,options={'object_count':count}); p.reset(s,info); start=time.monotonic()
  for step in range(env.max_steps):
   s,r,t,tr,info=env.step(p.get_action(s))
   if t or tr: break
  bot=s.get_objects(env.observation_space.get_type('mujoco_tidybot_robot'))[0]
  print(json.dumps({'seed':seed,'count':count,'steps':step+1,'term':t,'trunc':tr,'pos':[round(s.get(bot,f),3) for f in ['pos_base_x','pos_base_y']], 'seconds':round(time.monotonic()-start,2)}),flush=True)
