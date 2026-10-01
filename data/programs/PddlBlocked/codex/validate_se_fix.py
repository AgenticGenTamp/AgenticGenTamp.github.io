"""Full-episode validation of the staged, heading-relative southeast fix."""
from env_client import make_env
from approach import GeneratedApproach
import math
import numpy as np


def move(p, env, s, target, lift, arm=False, limit=30):
    for _ in range(limit):
        if p.at(s, target, arm=arm): return s, True
        s, _, done, truncated, _ = env.step(p.motion(s, target, 1, lift=lift, q4add=.08))
        if done or truncated: return s, False
    return s, False


for seed in [74,79,120,176,196]:
    env=make_env();s,info=env.reset(seed=seed)
    p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,info)
    steps=0
    while p.stage!=5 and steps<100:
        s,_,done,truncated,_=env.step(p.get_action(s));steps+=1
    # Make the q4 posture change while the arm is still outside the pen.
    for _ in range(2):
        a=np.zeros(11,np.float32);a[6]=np.clip(p.Q[3]+.08-p.g(s,'robot','joint_4'),-.2,.2);a[10]=1
        s,_,done,truncated,_=env.step(a);steps+=1
    c,z=math.cos(p.theta),math.sin(p.theta)
    local=np.array([.10,-.125])
    correction=np.array([c*local[0]-z*local[1],z*local[0]+c*local[1]])
    target=p.target(p.green)+correction
    corner=np.array([target[0],p.robot(s)[1]])
    for waypoint,lift,arm in ((corner,True,False),(target,True,False),(target,False,True)):
        s,ok=move(p,env,s,waypoint,lift,arm);steps+=30 if not ok else 1
    a=np.zeros(11,np.float32);a[10]=-1;s,_,done,truncated,_=env.step(a);steps+=1
    held=p.g(s,'robot','grasp_active')>.5
    # Pull horizontally through the gap before lifting; lifting inside the pen
    # is rejected for this shortened-arm grasp.
    outside=p.robot(s)+.62*p.out
    s,extracted=move(p,env,s,outside,False,False,40)
    # Now lift clear of the tabletop and hand off to the normal carry route.
    for _ in range(5):
        if abs(p.g(s,'robot','joint_2')-(p.Q[1]-.2))<.004:break
        s,_,done,truncated,_=env.step(p.motion(s,outside,1,lift=True,q4add=.08));steps+=1
    out=np.array([p.g(s,'green0','pose_x'),p.g(s,'green0','pose_y')])
    side=.72 if p.out[1]>=0 else -.72
    p.route=[np.array([out[0],side]),np.array([2.50,side])]
    if side>0:p.route.append(np.array([2.50,-.72]))
    p.carry_corner=np.array([2.50,-.72]);p.stage=8
    while not done and not truncated and steps<250:
        s,_,done,truncated,_=env.step(p.get_action(s));steps+=1
    print(seed,'held',held,'extract',extracted,'done',done,'steps',steps,'stage',p.stage)
    env.close()
