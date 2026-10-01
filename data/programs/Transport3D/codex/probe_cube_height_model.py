"""Continue the verified box-contact IK downward and test cube contacts."""
import math
import numpy as np
from scipy.optimize import least_squares

from env_client import make_env
from probe_grasp_structured import command, fk, JOINT_OFFSET, val


LO = np.array([0., -.35, -math.pi, -2.5, 0., -.87, math.pi/2])
HI = np.array([5.2, 2.06, -1.0836, .16, 5.2, 1.36, 8.3408])
QS = [
 np.array([.951305288,1.970876128,-1.493305392,-1.219847079,4.230377134,.474353059,6.00596593]),
 np.array([4.155821175,.426711007,-1.502109215,-1.900626345,1.884001343,.06090929,5.236141916]),
 np.array([4.170779777,1.703020948,-2.855671258,-.627929854,4.26973606,1.31947716,7.283258434]),
]
OFFS = [(-.799987478,-.347800459,2.104792961),
        (.126212298,-.083534270,-3.139703362),
        (-.428185141,.054943428,2.978131848)]


def candidates():
    q0 = QS[-1]
    t0 = fk(q0 + JOINT_OFFSET)
    out = []
    seed = q0.copy()
    for dz in np.arange(.02, .181, .02):
        target = t0.copy(); target[2, 3] -= dz
        def fun(q):
            t = fk(q + JOINT_OFFSET)
            # Match full transform locally; matrix entries avoid angle wrapping.
            return np.r_[12*(t[:3,3]-target[:3,3]),
                         2*(t[:3,:3]-target[:3,:3]).ravel(),
                         .03*(q-q0)]
        sol = least_squares(fun, seed, bounds=(LO, HI), max_nfev=1000)
        seed = sol.x
        out.append((dz, sol.x, np.linalg.norm(fun(sol.x))))
    return out


def run(seed=0):
    for dz, q, err in candidates():
        env = make_env(); state, _ = env.reset(seed=seed)
        tx, ty = val(state, 'cube0', 'pose_x'), val(state, 'cube0', 'pose_y')
        for qq, off in zip(QS, OFFS):
            state = command(env, state, [tx+off[0],ty+off[1],off[2]], qq, 1., 35)
        # Reach the continued posture, then compensate model x/y errors with a grid.
        b = np.array([tx+OFFS[-1][0], ty+OFFS[-1][1]])
        state = command(env, state, [b[0],b[1],OFFS[-1][2]], q, 1., 25)
        vals = np.arange(-.12, .121, .03)
        for iy, dy in enumerate(vals):
            xs = vals if iy % 2 == 0 else vals[::-1]
            for dx in xs:
                bb = [b[0]+dx,b[1]+dy,OFFS[-1][2]]
                state = command(env,state,bb,q,1.,3)
                state = command(env,state,bb,q,-1.,2)
                if val(state,'robot','grasp_active') > .5:
                    actual=[val(state,'robot','joint_%d'%i) for i in range(1,8)]
                    print('HIT dz',dz,'err',err,'q',q.tolist(),'actual',actual,
                          'baseoff',val(state,'robot','pos_base_x')-tx,
                          val(state,'robot','pos_base_y')-ty,flush=True)
                    env.close(); return
        print('MISS dz',dz,'err',err,'q',np.round(q,5).tolist(),flush=True)
        env.close()


if __name__ == '__main__':
    run()
