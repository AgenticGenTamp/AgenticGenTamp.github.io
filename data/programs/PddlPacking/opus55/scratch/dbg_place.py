import sys; sys.path.insert(0,'.')
import numpy as np, math
from env_client import make_env
from approach import *
from kin import fk_full
seed=int(sys.argv[1]); N=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(N):
    a=ap.get_action(obs); obs,*_=env.step(a)
cfg,blocks,holding,held,gtf,gq=ap._parse(obs)
print("holding",holding,held, cfg[:3].round(2))
heldinfo=(gtf,quat_to_R(gq))
w2=World({n:b for n,b in blocks.items() if n!=held})
placed=[b for n,b in blocks.items() if n!=held and ap._on_plate(b)]
for ex in (False,True):
    o=ap._place_options(cfg,w2,heldinfo,placed,0,extra=ex)
    print("extra",ex,[(s,c[:3].round(2).tolist()) for s,c in o[:6]])
# manual: bases on -y side
for bx in (-0.15,0.0,0.15):
  for sl in [(0.0,-0.094),(0.06,-0.06)]:
    r=ap._place_ik((bx,-0.99,0.0),cfg,sl,heldinfo[0],heldinfo[1],False,w2,heldinfo)
    r2=ap._place_ik((bx,-0.99,0.0),cfg,sl,heldinfo[0],heldinfo[1],True,w2,heldinfo)
    print(bx,sl, r is not None, r2 is not None)
