from env_client import make_env
import numpy as np

def snapshot(s):
    return {n:{f:round(s.get(s.get_object_from_name(n),f),5) for f in ['x','y','theta']+(['arm_joint','arm_length','vacuum','gripper_height','gripper_width','base_radius'] if n=='robot' else ['width','height','static'])} for n in s.get_object_names()}

def run(seed=0):
    e=make_env();s,_=e.reset(seed=seed);print('INIT',snapshot(s),'limit',e.max_steps)
    for label,a,num in [('armout',[0,0,0,.1,0],10),('right',[.05,0,0,0,0],3),('rotate',[0,0,.196,0,0],4),('armin',[0,0,0,-.1,0],10),('down',[0,-.05,0,0,0],8)]:
        for i in range(num):
            s,r,t,tr,info=e.step(np.array(a));print(label,i,snapshot(s)['robot'],r,t,tr,info)
    e.close()
if __name__=='__main__':run()
