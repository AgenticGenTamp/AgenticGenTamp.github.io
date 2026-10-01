"""Explore coupled lowering/restoration at the shallow-east green grasp."""
import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach


def g(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def setup(seed):
    e = make_env(); s, info = e.reset(seed=seed)
    p = GeneratedApproach(e.action_space, e.observation_space, {}); p.reset(s, info)
    for _ in range(100):
        if p.stage == 5:
            break
        s, *_ = e.step(p.get_action(s))
    # Run only the lifted base-positioning portion of stage 5.
    target = None
    for _ in range(30):
        old = p.robot(s).copy(); a = p.get_action(s)
        if p.stage != 5:
            break
        s2, *_ = e.step(a)
        if np.max(np.abs(p.robot(s2)-old)) < 1e-7:
            # This is the first rejected arm-lowering action.
            target = p.robot(s).copy(); break
        s = s2
    return e, s, p, target


def delta(e, s, entries, grip=1.):
    a = np.zeros(11, np.float32); a[10] = grip
    for i, x in entries.items(): a[i] = np.clip(x, -.2, .2)
    return e.step(a)[0]


def move_relative(e, s, index, amount):
    left = amount
    while abs(left) > 1e-5:
        d = np.clip(left, -.2, .2); before = g(s,'robot','joint_'+str(index-2))
        s = delta(e, s, {index:d}); after = g(s,'robot','joint_'+str(index-2))
        if abs(after-before) < 1e-7: return s, False
        left -= d
    return s, True


def run(seed, mode, n=20):
    e,s,p,_=setup(seed)
    try:
        q0=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
        s,ok5=move_relative(e,s,7,1.4)
        s,ok7=move_relative(e,s,9,.4)
        start=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
        accepted=0
        # Desired net changes: q2 +.2, q5 -1.4, q7 -.4.
        for k in range(n):
            u=(k+1)/n
            if mode=='linear': f2=f5=f7=u
            elif mode=='lower_first': f2=min(1.,2*u); f5=f7=max(0.,2*u-1.)
            elif mode=='restore_first': f5=f7=min(1.,2*u); f2=max(0.,2*u-1.)
            elif mode=='q5_late': f2=f7=u; f5=u*u
            elif mode=='q7_late': f2=f5=u; f7=u*u
            elif mode=='q5_early': f2=f7=u; f5=2*u-u*u
            elif mode=='q7_early': f2=f5=u; f7=2*u-u*u
            goals={4:q0[1]+.2*f2,7:start[4]-1.4*f5,9:start[6]-.4*f7}
            entries={i:goals[i]-g(s,'robot','joint_'+str(i-2)) for i in goals}
            before=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
            s=delta(e,s,entries)
            after=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
            accepted += int(np.max(abs(after-before))>1e-7)
        pre=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
        s=delta(e,s,{},-1.)
        held=g(s,'robot','grasp_active')
        print(seed,mode,'rollok',ok5,ok7,'accepted',accepted,'q2/5/7',
              np.round(pre[[1,4,6]],4).tolist(),'held',held)
        return held
    finally:e.close()

def underpass(seed, low, order):
    e,s,p,_=setup(seed)
    try:
        q0=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
        s,_=move_relative(e,s,7,1.4);s,_=move_relative(e,s,9,.4)
        # Lower shoulder past nominal while the wrist is folded clear.
        for goal_index,goal in ((4,low),):
            for _ in range(30):
                cur=g(s,'robot','joint_2');d=np.clip(goal-cur,-.05,.05)
                if abs(d)<1e-4:break
                s=delta(e,s,{goal_index:d})
        # Restore either roll first or both simultaneously at the low posture.
        targets={7:q0[4],9:q0[6]}
        for idx in order:
            for _ in range(40):
                d=p.w(targets[idx]-g(s,'robot','joint_'+str(idx-2)))
                if abs(d)<1e-4:break
                s=delta(e,s,{idx:np.clip(d,-.05,.05)})
        for _ in range(40):
            entries={idx:p.w(targets[idx]-g(s,'robot','joint_'+str(idx-2))) for idx in targets}
            entries[4]=q0[1]+.2-g(s,'robot','joint_2')
            if max(abs(x) for x in entries.values())<1e-4:break
            s=delta(e,s,{i:np.clip(x,-.05,.05) for i,x in entries.items()})
        pre=np.array([g(s,'robot','joint_'+str(i)) for i in range(1,8)])
        s=delta(e,s,{},-1.);held=g(s,'robot','grasp_active')
        print('UNDER',seed,low,order,'q2/5/7',np.round(pre[[1,4,6]],4).tolist(),'held',held)
        return held
    finally:e.close()


if __name__=='__main__':
    seeds=[101,191] if len(sys.argv)<2 else [int(sys.argv[1])]
    for seed in seeds:
        for mode in ('linear','lower_first','restore_first','q5_late','q7_late','q5_early','q7_early'):
            run(seed,mode,30)
        for low in (.3,.4,.5,.6,.8,1.0,1.2):
            for order in ((7,9),(9,7)):
                underpass(seed,low,order)
