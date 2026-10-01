import numpy as np
LIM = np.array([0.049, 0.049, 0.19, 0.099, 0.0199])
def rob(obs):
    R = obs.get_object_from_name('robot')
    return np.array([obs.get(R,f) for f in ['x','y','theta','arm_joint','finger_gap']])
def objs(obs):
    s={}
    for o in sorted(obs.data,key=lambda o:o.name):
        if o.name!='robot': s[o.name]=tuple(round(obs.get(o,f),3) for f in ['x','y','theta','held'])
    return s
def goto(env, obs, target, maxsteps=200, tol=1e-3, verbose=False):
    """target: array of 5 (x,y,theta,arm,gap), None entries kept"""
    for i in range(maxsteps):
        r = rob(obs)
        t = np.array([r[k] if target[k] is None else target[k] for k in range(5)])
        d = t - r
        d[2] = (d[2]+np.pi)%(2*np.pi)-np.pi
        if np.all(np.abs(d) < tol): break
        a = np.clip(d, -LIM, LIM)
        obs, rew, term, trunc, info = env.step(a)
        if verbose: print(rob(obs).round(3), objs(obs))
        if term: print("TERMINATED"); return obs, True
    return obs, False
