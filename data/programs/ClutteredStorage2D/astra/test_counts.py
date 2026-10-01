from env_client import make_env
from approach import GeneratedApproach
import time
for count,seed in [(19,0),(19,1)]:
 env=make_env();state,info=env.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(env.action_space,env.observation_space,{})
 start=time.monotonic();p.reset(state,info);term=False;compute=0.;maxaction=0.;lastfinished=set()
 for i in range(env.max_steps):
  ast=time.monotonic();a=p.get_action(state);dt=time.monotonic()-ast;compute+=dt;maxaction=max(maxaction,dt)
  state,reward,term,trunc,info=env.step(a)
  if p.finished!=lastfinished:
   print("MILESTONE",count,seed,"STEP",i+1,"NEW",sorted(p.finished-lastfinished),"COMPUTE",round(compute,2),flush=True);lastfinished=set(p.finished)
  if term or trunc or time.monotonic()-start>78:break
 print('COUNT',count,'SEED',seed,'OK',term,'STEPS',i+1,'SEC',round(time.monotonic()-start,2),'COMPUTE',round(compute,3),'MAXACT',round(maxaction,3),'PHASE',p.phase,'TARGET',p.target,'Q',[round(float(x),3) for x in p.q],flush=True)
 if not term:print('BLOCKS',{k:[round(float(x),3) for x in v] for k,v in p.blocks.items()},'FINISHED',p.finished,'FAILURES',p.failures,flush=True)
 env.close()
 if not term:break
