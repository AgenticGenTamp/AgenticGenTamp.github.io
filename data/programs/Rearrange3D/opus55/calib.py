import numpy as np, sys
from multiprocessing import Pool
from env_client import make_env
import approach
from approach import GeneratedApproach
def run(args):
    seed,fd,fc=args
    approach.FORCE.clear()
    if fd is not None: approach.FORCE[16]=fd
    if fc is not None: approach.FORCE[32]=fc
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for t in range(600):
        obs,r,te,tr,info=env.step(ap.get_action(obs))
        if te or tr: break
    b=obs[0:2]
    return (seed,fd,fc,te,t,(obs[16:18]-b).round(3),(obs[32:34]-b).round(3),obs[19].round(2),obs[22].round(2))
if __name__=='__main__':
    seed=int(sys.argv[1])
    jobs=[]
    for r in (0.10,0.115,0.13):
        for a in (90,135):
            ang=np.radians(a); jobs.append((seed,(r*np.cos(ang),r*np.sin(ang)),(0.0,-0.10)))
    for r in (0.12,0.14,0.16):
        jobs.append((seed,(0.0,0.10),(0.0,-r)))
    with Pool(9) as p:
        for x in p.map(run,jobs): print(x)
