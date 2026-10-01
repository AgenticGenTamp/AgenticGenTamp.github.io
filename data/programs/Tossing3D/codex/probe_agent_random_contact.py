"""Reproducible broad random contact search; never imported by approach."""
import numpy as np
from env_client import make_env

def vals(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([float(s.get(o, f)) for f in fs])

def main():
    rng = np.random.RandomState(731991)
    env = make_env(); s, _ = env.reset(seed=0, options={"object_count": 1})
    fs = tuple("pos_arm_joint%d" % i for i in range(1, 8))
    p0 = vals(s, "cube_0", ("x", "y", "z")); actions=[]
    # Put the cube in a dense range of possible arm workspaces by changing
    # base/cube relative pose every 60 steps. Random walk the joints strongly.
    offsets = [(x,y) for x in (.15,.3,.45,.6,.75) for y in (-.45,-.25,0,.25,.45)]
    rng.shuffle(offsets)
    target_q = vals(s,"robot",fs)
    for t in range(980):
        block = min(t // 38, len(offsets)-1)
        off = np.array(offsets[block]); base_goal=p0[:2]-off
        base=vals(s,"robot",("pos_base_x","pos_base_y")); q=vals(s,"robot",fs)
        if t % 7 == 0:
            # Broad legal-looking Gen3 configurations; a quick delta controller
            # takes us there while the base remains close enough for collision.
            target_q = rng.uniform(-3.0,3.0,7)
        a=np.zeros(18,np.float32)
        a[:2]=np.clip(base_goal-base,-.1,.1)
        a[2]=rng.uniform(-.1,.1)
        a[3:10]=np.clip(target_q-q,-.1,.1)
        a[10]=float((t//5)%2)
        a[11:18]=np.clip(8*(target_q-q),-12,12)
        actions.append(a.copy())
        s,r,term,trunc,info=env.step(a)
        p=vals(s,"cube_0",("x","y","z")); v=vals(s,"cube_0",("vx","vy","vz"))
        d=float(np.linalg.norm(p-p0))
        if d > 2e-4 or np.linalg.norm(v)>2e-3:
            print("FOUND t",t,"delta",p-p0,"vel",v,"off",off,"q",q,"action",a,flush=True)
            np.savez("random_contact_found.npz", actions=np.array(actions), p0=p0)
            env.close(); return
        if term or trunc: break
    print("NO CONTACT",len(actions),flush=True); env.close()
if __name__ == "__main__": main()
