import sys
from env_client import make_env
from approach import GeneratedApproach

seed=int(sys.argv[1])
env=make_env(); s,info=env.reset(seed=seed,options={"object_count":int(sys.argv[2]) if len(sys.argv)>2 else 0})
p=GeneratedApproach(env.action_space,env.observation_space,{}) ; p.reset(s,info)
t={q.name:q for q in env.observation_space.types}
for i in range(1000):
    s,r,d,tr,inf=env.step(p.get_action(s))
    if i%20==0 or (25 <= i <= 65) or (115 <= i <= 150) or d:
        ro=list(s.get_objects(t['crv_robot']))[0]; b=list(s.get_objects(t['target_block']))[0]; g=list(s.get_objects(t['target_region']))[0]
        v=lambda o,f:round(float(s.get(o,f)),3)
        print(i,p.phase,p.name,'tries',dict(p.attempts),'cleared',len(p.cleared),'rob',v(ro,'x'),v(ro,'y'),v(ro,'theta'),v(ro,'arm_joint'),v(ro,'vacuum'),'b',v(b,'x'),v(b,'y'),v(b,'theta'),'g',v(g,'x'),v(g,'y'),v(g,'theta'), 'delta', float(s.get(b,'x')-s.get(g,'x')),float(s.get(b,'y')-s.get(g,'y')),d)
        if len(sys.argv)>2:
            oo=s.get_object_from_name('obstruction0'); print(' obs',v(oo,'x'),v(oo,'y'),v(oo,'theta'))
    if d or tr: break
    if p.phase == "finish":
        print("RAW", {f:float(s.get(b,f)) for f in ('x','y','theta','width','height')},
              {f:float(s.get(g,f)) for f in ('x','y','theta','width','height')},
              {f:float(s.get(ro,f)) for f in ('x','y','theta','vacuum','arm_joint')})
        break
env.close()
