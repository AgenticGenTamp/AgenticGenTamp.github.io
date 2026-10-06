import sys; sys.path.insert(0,'/sandbox/scratch'); from h import *
oc=env._kinder_env._object_centric_env
h=H(32); h.step(60)
o=h.obs; print("bowl",np.round(o[0:13],3)); print("8-30",np.round(o[13:45],3))
print([ (k,type(v).__name__) for k,v in list(oc.__dict__.items())[:60] if 'obj' in k or 'fix' in k or 'task' in k])
print(oc.task_config["regions"], oc.task_config["goal_state"])
