from env_client import make_env
from approach import GeneratedApproach
import sys,time,json,argparse
parser=argparse.ArgumentParser()
parser.add_argument("counts",nargs="*",type=int,default=[1,2,5,10,20,50])
parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
args=parser.parse_args()
for n in args.counts:
 for seed in args.seeds:
  E=make_env();s,i=E.reset(seed=seed,options={'object_count':n});p=GeneratedApproach(E.action_space,E.observation_space,{})
  start=time.monotonic();p.reset(s,i);compute=time.monotonic()-start
  for k in range(E.max_steps):
   t=time.monotonic();a=p.get_action(s);compute+=time.monotonic()-t
   s,r,done,trunc,i=E.step(a)
   if done or trunc or compute>55:break
  remaining=[o.name for o in s.get_objects(E.observation_space.get_type('circle')) if s.get(o,'color_g')<.5]
  print(json.dumps({'count':n,'seed':seed,'steps':k+1,'success':done,'remaining':remaining,'phase':p.phase,'compute':round(compute,3),'time':round(time.monotonic()-start,3)}),flush=True)
  E.close()
