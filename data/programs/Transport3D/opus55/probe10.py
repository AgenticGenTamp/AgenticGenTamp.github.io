from util import *
e=E(1)
c=e.o('cube0')
e.base_to(c['pose_x']-0.6, c['pose_y'])
e.ee_to([0.5,0,-0.1]); e.ee_to([0.5,0,-0.2])
def pr(t):
    r=e.rob(); print(t,'fs',round(r['finger_state'],3),'ga',r['grasp_active'],[round(r[k],4) for k in RF[12:]], {k:round(v,4) for k,v in e.o('cube0').items()})
pr('before')
for i in range(4): e.gripper(-1); pr('close')
for i in range(2): e.step(np.zeros(11)); pr('noop')
print(e.ee_to([0.5,0,-0.1])); pr('lift')
print(e.base_to(c['pose_x']-0.4, c['pose_y'])); pr('base')
for i in range(4): e.gripper(1); pr('open')
print(e.ee_to([0.5,0,0.0])); pr('lift')
