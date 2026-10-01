"""Find alternate wrist branches that evade live self-collision constraints."""
import numpy as np
from scipy.optimize import minimize_scalar
from env_client import make_env
from probe_grasp_sequence import drive, value
from solve_alt_fk import fk

env = make_env()
rows = []
for q4 in (-.6, -1.0, -1.3):
    for q3 in (0.0, np.pi/2, np.pi, -np.pi/2):
        for q5 in (0.0, np.pi/2, -np.pi/2):
            def orient(q6):
                return np.sum((fk([0,2.24,q3,q4,q5,q6,1.57])[:3,2]-[0,0,-1])**2)
            q6 = minimize_scalar(orient, bounds=(-2.1,2.1), method="bounded").x
            state, _ = env.reset(seed=0, options={"object_count": 1})
            target = np.array([0,2.24,q3,q4,q5,q6,1.57])
            state = drive(env,state,target,0.0,120)
            actual = np.array([value(state,"robot",f"pos_arm_joint{i}") for i in range(1,8)])
            tool = fk(actual)
            score = tool[2,3] + .15*np.sum((tool[:3,2]-[0,0,-1])**2)
            rows.append((score, target, actual, tool[:3,3], tool[:3,2]))
for score,target,actual,p,axis in sorted(rows,key=lambda x:x[0])[:12]:
    print("score",round(score,3),"target",np.round(target,2),"actual",np.round(actual,2),
          "p",np.round(p,3),"axis",np.round(axis,2))
env.close()
