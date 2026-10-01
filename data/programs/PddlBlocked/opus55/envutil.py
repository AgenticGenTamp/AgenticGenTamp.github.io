import numpy as np
JN=['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']
class Sim:
    def __init__(self, env, obs):
        self.env=env; self.obs=obs; self.T={t.name:t for t in env.observation_space.type_features}
        self.r=obs.get_objects(self.T['robot'])[0]; self.steps=0; self.done=False
    def rget(self,f): return float(self.obs.get(self.r,f))
    def base(self): return np.array([self.rget('base_x'),self.rget('base_y'),self.rget('base_rot')])
    def q(self): return np.array([self.rget(j) for j in JN])
    def block(self,name):
        o=self.obs.get_object_from_name(name)
        return np.array([float(self.obs.get(o,f)) for f in ['pose_x','pose_y','pose_z','pose_qx','pose_qy','pose_qz','pose_qw','grasp_active']])
    def step(self,a):
        a=np.asarray(a,dtype=np.float32)
        self.obs,rew,term,trunc,info=self.env.step(a); self.steps+=1; self.done=term
        return term
    def moveto(self, base, q, grip=0.0, maxd=0.2):
        """Interpolate to (base,q). returns True if reached, False if blocked."""
        for _ in range(200):
            b=self.base(); qq=self.q()
            db=np.array(base)-b; db[2]=(db[2]+np.pi)%(2*np.pi)-np.pi
            dq=np.array(q)-qq; dq[4]=(dq[4]+np.pi)%(2*np.pi)-np.pi; dq[6]=(dq[6]+np.pi)%(2*np.pi)-np.pi
            delta=np.concatenate([db,dq])
            m=np.abs(delta).max()
            if m<1e-4: return True
            a=delta/max(1.0,m/maxd)
            a=np.concatenate([a,[grip]])
            prev=np.concatenate([b,qq])
            self.step(a)
            if self.done: return True
            now=np.concatenate([self.base(),self.q()])
            if np.abs(now-prev).max()<1e-6:
                return False
        return False
