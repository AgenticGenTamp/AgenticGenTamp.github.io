import sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); lim=int(sys.argv[2]) if len(sys.argv)>2 else 60
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
print("plan",ap.plan, "pillars",[tuple(round(v,2) for v in p[:2]) for p in ap.pillars])
print("objs",[tuple(round(v,2) for v in p) for p in ap.obj_xy])
print("samples",[(k,round(x,2),round(y,2),s) for k,((x,y),s) in enumerate(zip(ap.samp_xy,ap.samp_soil))])
g=obs.get
for t in range(lim):
    a=ap.get_action(obs)
    pre=[(round(g(ap.rover_objs[r],'x'),2),round(g(ap.rover_objs[r],'y'),2)) for r in ap.rnames]
    obs,r,term,trunc,_=env.step(a); g=obs.get
    st=[(round(g(ap.rover_objs[r],'x'),2),round(g(ap.rover_objs[r],'y'),2),int(g(ap.rover_objs[r],'store_full')),int(g(ap.rover_objs[r],'calibrated')),int(g(ap.rover_objs[r],'at_home'))) for r in ap.rnames]
    ob=''.join(str(int(g(o,'have_image_rover0')))+str(int(g(o,'have_image_rover1')))+str(int(g(o,'received_image')))+' ' for o in ap.objectives)
    sm=''.join(str(int(g(o,'analyzed_rover0')))+str(int(g(o,'analyzed_rover1')))+str(int(g(o,'received_analysis')))+' ' for o in ap.samples)
    ops=[round(float(a[3]),2),round(float(a[7]),2)]
    print(t,ops,[ap.targets[r] for r in ap.rnames],st,ob,'|',sm)
    if term: print("TERM");break
