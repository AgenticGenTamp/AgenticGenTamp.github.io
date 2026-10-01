import math
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space = observation_space
        self.low = np.asarray(action_space.low) + 1e-7
        self.high = np.asarray(action_space.high) - 1e-7
        self.rt = self.space.get_type('kin_robot')
        self.ht = self.space.get_type('hook')
        self.tt = self.space.get_type('target_block')
    def reset(self, state, info):
        self.t = 0
        self.phase = 0
    def move(self, state, robot, x, y, theta, joint=.24, gap=.32):
        angle = (theta-state.get(robot,'theta')+math.pi)%(2*math.pi)-math.pi
        return np.clip([x-state.get(robot,'x'),y-state.get(robot,'y'),angle,
                        joint-state.get(robot,'arm_joint'),gap-state.get(robot,'finger_gap')],self.low,self.high)
    def get_action(self, state):
        self.t += 1
        robot = state.get_objects(self.rt)[0]
        hook = state.get_objects(self.ht)[0]
        hx,hy,ht = [state.get(hook,f) for f in ('x','y','theta')]
        length=state.get(hook,'length_side1')
        width=state.get(hook,'width')
        x=hx-(length+.24)*math.cos(ht)+width/2*math.sin(ht)
        y=hy-(length+.24)*math.sin(ht)-width/2*math.cos(ht)
        dist=math.hypot(x-state.get(robot,'x'),y-state.get(robot,'y'))
        gap=.12 if dist<.03 else .32
        if state.get(hook,'held'):
            self.phase=1
            return self.move(state,robot,state.get(robot,'x'),1.4,math.pi/2,gap=.12)
        return self.move(state,robot,x,y,ht,gap=gap)
