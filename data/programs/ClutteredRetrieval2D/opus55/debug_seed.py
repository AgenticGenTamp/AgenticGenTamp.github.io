import sys, time
from env_client import make_env
import approach; approach.DEBUG=True
from approach import GeneratedApproach
seed = int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info); t0=time.time()
orig = ap._plan
def plan():
    t=time.time(); orig()
    print(f"step {step} plan held={ap.held} q={ap.q.round(3)} queue={len(ap.queue)} pending={ap.pending} tgt={getattr(ap,'grasp_target',None)} dt={time.time()-t:.2f}")
ap._plan = plan
for step in range(1000):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(a)
    if term: print("SOLVED", step+1); break
    if time.time()-t0 > 70: print("TIMEOUT"); break
