from kinder.envs.kinematic2d.clutteredretrieval2d import ClutteredRetrieval2DEnv
import inspect
e=ClutteredRetrieval2DEnv(num_obstructions=3)
s,_=e.reset(seed=0)
c=e.initial_constant_state
for o in c: print(o.name,[round(float(c.get(o,f)),3) for f in ['x','y','theta','width','height','z_order','static']])
print(type(e).__mro__)
print([a for a in dir(e) if not a.startswith('__')][:80])
