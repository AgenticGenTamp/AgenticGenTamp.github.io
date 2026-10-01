import math
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space = observation_space
        self.low = action_space.low * .999
        self.high = action_space.high * .999

    def reset(self, state, info):
        self.phase = 'prepare'
        self.objname = None
        self.cleared = set()
        self.steps = 0

    def get_action(self, state):
        self.steps += 1
        r = next(iter(state.get_objects(self.space.get_type('kin_robot'))))
        g = lambda o, f: state.get(o, f)
        surf = next(iter(state.get_objects(self.space.get_type('target_surface'))))
        target = next(iter(state.get_objects(self.space.get_type('target_block'))))
        if self.objname is None:
            sx, sw = g(surf, 'x'), g(surf, 'width')
            blocks = list(state.get_objects(self.space.get_type('dyn_rectangle')))
            blockers = [o for o in blocks if abs(g(o,'x')-sx)<(sw+g(o,'width'))/2+.05 and -.1<g(o,'y')<1]
            obj = min(blockers,key=lambda o: g(o,'x')) if blockers else target
            self.objname = obj.name
            self.phase = 'prepare'
            self.pick = (g(obj,'x'),g(obj,'y')+.50)
        obj = state.get_object_from_name(self.objname)
        ox,oy = g(obj,'x'),g(obj,'y')
        theta = -math.pi/2
        arm = .48
        dist = .50
        gap = min(.65,g(obj,'width')+.10)
        x,y = self.pick
        if self.phase == 'prepare':
            y = max(1.5,oy+dist+.4)
            if abs(g(r,'x')-x)<.03 and abs(g(r,'y')-y)<.03 and abs(g(r,'theta')-theta)<.03:
                self.phase='descend'
        elif self.phase == 'descend':
            if abs(g(r,'x')-x)<.015 and abs(g(r,'y')-y)<.015:
                self.phase='grasp'
        elif self.phase == 'grasp':
            gap=0
            if g(r,'finger_gap')<.121 and not g(obj,'held'):
                self.pick=(x,y-.025)
            if oy<-.2:
                self.objname=None
            if g(obj,'held'):
                self.phase='lift'
                self.offset=(g(r,'x')-ox,g(r,'y')-oy)
        elif self.phase == 'lift':
            gap=0
            y=1.65
            if g(r,'y')>1.62:
                self.phase='carry'
        elif self.phase == 'carry':
            gap=0
            x=(g(surf,'x') if obj==target else (.5 if g(surf,'x')>1.8 else 3.3))+self.offset[0]
            y=1.65
            if abs(g(r,'x')-x)<.02:
                self.phase='lower'
        elif self.phase == 'lower':
            gap=0
            x=(g(surf,'x') if obj==target else (.5 if g(surf,'x')>1.8 else 3.3))+self.offset[0]
            y=.1+g(obj,'height')/2+self.offset[1]+.04
            if abs(g(r,'y')-y)<.02:
                self.phase='release'
        elif self.phase == 'release':
            gap=.65
            if not g(obj,'held'):
                self.objname=None
        a=[x-g(r,'x'),y-g(r,'y'),theta-g(r,'theta'),.24-g(r,'arm_joint'),gap-g(r,'finger_gap')]
        return np.clip(a,self.low,self.high).astype(np.float32)
