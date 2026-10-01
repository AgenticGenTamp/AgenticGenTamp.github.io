from env_client import make_env
from approach import GeneratedApproach
import numpy as np, sys

for seed in map(int, sys.argv[1:]):
    e=make_env(); s,info=e.reset(seed=seed); p=GeneratedApproach(e.action_space,e.observation_space,{})
    p.reset(s,info); last_phase=-1; last_button=s[20:22].copy(); print('\nSEED',seed)
    for k in range(700):
        a=p.get_action(s); s,r,t,tr,info=e.step(a)
        moved=np.linalg.norm(s[20:22]-last_button)
        if p.phase!=last_phase or moved>.004 or k in (200,400,699):
            u=np.array([np.cos(s[11]),np.sin(s[11])]); n=np.array([u[1],-u[0]])
            print(k,'ph',p.phase,'r',np.round(s[:2],2),'h',np.round(s[9:11],2),'tip',np.round(s[9:11]+s[19]*n,2),'b',np.round(s[20:22],2),'d',round(float(np.linalg.norm(s[20:22]-s[29:31])),3),'a',np.round(a[:2],2),'el',getattr(p,'use_elbow',None))
            last_phase=p.phase; last_button=s[20:22].copy()
        if t or tr: break
    e.close()
