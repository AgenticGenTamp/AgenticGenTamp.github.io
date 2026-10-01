import math
import heapq
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space = observation_space
        self.low = np.asarray(action_space.low)
        self.high = np.asarray(action_space.high)
        self.robot_type = observation_space.get_type('crv_robot')
        self.rect_type = observation_space.get_type('rectangle')
        self.target_type = observation_space.get_type('target_region')

    def reset(self, state, info):
        self.robot = next(iter(state.get_objects(self.robot_type)))
        self.target = next(iter(state.get_objects(self.target_type)))
        self.radius = float(state.get(self.robot, 'base_radius'))
        self.boxes = []
        for obj in state.get_objects(self.rect_type):
            if obj == self.target:
                continue
            x,y,w,h,t = [float(state.get(obj,f)) for f in ('x','y','width','height','theta')]
            c,s = math.cos(t),math.sin(t)
            corners = np.array([[0,0],[w,0],[0,h],[w,h]]) @ np.array([[c,s],[-s,c]]) + [x,y]
            lo,hi = corners.min(axis=0),corners.max(axis=0)
            margin = np.array([self.radius+.015,self.radius+.001])
            self.boxes.append(np.r_[lo-margin,hi+margin])
        self.boxes = np.asarray(self.boxes).reshape(-1,4)
        self.margin = np.array([self.radius+.015,self.radius+.001])
        columns = sorted(set(round(float(x),6) for x in self.boxes[:,0]))
        self.vertical = any(b-a < 2*self.radius+.03 for a,b in zip(columns,columns[1:]))
        if self.vertical:
            new_margin = np.array([self.radius+.0001,self.radius+.006])
            self.boxes[:,:2] += self.margin-new_margin
            self.boxes[:,2:] -= self.margin-new_margin
            self.margin = new_margin
        self.goal = np.array([state.get(self.target,'x'),state.get(self.target,'y')],float)
        self.goal_size = np.array([state.get(self.target,'width'),state.get(self.target,'height')],float)
        self.path = []
        self.previous = None
        self.stuck = 0
        self.calls = 0
        theta = float(state.get(self.robot, 'theta'))
        axis = math.pi/2 if self.vertical else 0.
        self.orientation = axis+round((theta-axis)/math.pi)*math.pi

    def clear(self, a, b):
        if len(self.boxes) == 0:
            return True
        d = b-a
        lo = np.full(len(self.boxes),-np.inf)
        hi = np.full(len(self.boxes),np.inf)
        for k in range(2):
            if abs(d[k]) < 1e-10:
                inside = (a[k] > self.boxes[:,k]+1e-8) & (a[k] < self.boxes[:,k+2]-1e-8)
                lo[~inside] = np.inf
            else:
                u = (self.boxes[:,k]+1e-8-a[k])/d[k]
                v = (self.boxes[:,k+2]-1e-8-a[k])/d[k]
                lo = np.maximum(lo,np.minimum(u,v))
                hi = np.minimum(hi,np.maximum(u,v))
        return not np.any(np.maximum(lo,0) < np.minimum(hi,1))

    def plan(self, pos):
        eps = .0001
        goals = []
        for fx in (eps,.5,1-eps):
            for fy in (eps,.5,1-eps):
                goals.append(self.goal+self.goal_size*np.array([fx,fy]))
        goals.insert(0,np.clip(pos,self.goal+eps,self.goal+self.goal_size-eps))
        nodes = [pos]+goals
        for x0,y0,x1,y1 in self.boxes:
            nodes.extend((np.array([x0,y0]),np.array([x0,y1]),np.array([x1,y0]),np.array([x1,y1])))
        extra_goals = [np.clip(p,self.goal+eps,self.goal+self.goal_size-eps) for p in nodes[1+len(goals):]]
        corners = nodes[1+len(goals):]
        goals = list({tuple(g):g for g in goals+extra_goals}.values())
        nodes = np.asarray([pos]+goals+corners)
        valid = np.all((nodes >= self.radius+.001) & (nodes <= 2.5-self.radius-.001),axis=1)
        valid[:1+len(goals)] = True
        n = len(nodes)
        dist = np.full(n,np.inf); dist[0] = 0
        prev = np.full(n,-1,dtype=int)
        queue = [(0.,0)]; visited = set()
        end = None
        while queue:
            cost,i = heapq.heappop(queue)
            if i in visited: continue
            visited.add(i)
            if 1 <= i <= len(goals):
                end = i; break
            distance = np.max(np.abs(nodes-nodes[i]),axis=1)
            ds = np.ceil(distance/.05-1e-6)+distance*1e-5
            for j in np.flatnonzero(valid & (cost+ds < dist-1e-9)):
                if j in visited: continue
                if self.clear(nodes[i],nodes[j]):
                    dist[j] = cost+ds[j]; prev[j] = i
                    heapq.heappush(queue,(dist[j],int(j)))
        if end is None:
            return [self.goal+self.goal_size/2]
        path = []
        while end != 0:
            path.append(nodes[end]); end = prev[end]
        return path[::-1]

    def get_action(self, state):
        self.calls += 1
        pos = np.array([state.get(self.robot,'x'),state.get(self.robot,'y')],float)
        theta = float(state.get(self.robot,'theta'))
        da = (self.orientation-theta+math.pi)%(2*math.pi)-math.pi
        action = np.array([0.,0.,np.clip(da,self.low[2],self.high[2]),-.1,0.],dtype=np.float32)
        if self.previous is not None and np.linalg.norm(pos-self.previous)<1e-6:
            self.stuck += 1
        else:
            self.stuck = 0
        if not self.path:
            self.path = self.plan(pos)
        while len(self.path)>1 and np.max(np.abs(pos-self.path[0])) < 1e-5:
            self.path.pop(0)
        for j in range(len(self.path)-1,0,-1):
            if self.clear(pos,self.path[j]):
                self.path = self.path[j:]
                break
        delta = self.path[0]-pos
        if self.stuck > 2:
            self.path = self.plan(pos)
            delta = self.path[0]-pos
        delta = delta/max(1.,np.max(np.abs(delta))/.05)
        # Spend remaining movement near a corner on the following segment.
        if len(self.path)>1 and np.max(np.abs(self.path[0]-pos))<=.05+1e-8:
            a,b = self.path[:2]
            d = b-a
            lower,upper = 0.,1.
            for k in range(2):
                if abs(d[k])<1e-10:
                    if abs(a[k]-pos[k])>.05: upper = -1.
                else:
                    t0,t1 = (pos[k]-.05-a[k])/d[k],(pos[k]+.05-a[k])/d[k]
                    lower = max(lower,min(t0,t1))
                    upper = min(upper,max(t0,t1))
            if lower<=upper and self.clear(pos,a+lower*d):
                if not self.clear(pos,a+upper*d):
                    for _ in range(16):
                        middle = (lower+upper)/2
                        if self.clear(pos,a+middle*d): lower = middle
                        else: upper = middle
                    upper = lower
                candidate = a+upper*d
                if upper>1e-6:
                    delta = candidate-pos
        if abs(da) > .001 and len(self.boxes):
            # While turning, the retracted gripper fits in this circular bound.
            p = pos+delta
            physical_lo = self.boxes[:,:2]+self.margin
            physical_hi = self.boxes[:,2:]-self.margin
            near = np.clip(p,physical_lo,physical_hi)
            turning_radius = max(self.radius,math.hypot(.105,.035))+.0001
            if np.any(np.sum((near-p)**2,axis=1)<turning_radius**2):
                delta[:] = 0.
            if np.any(p<turning_radius) or np.any(p>2.5-turning_radius):
                delta[:] = 0.
        action[:2] = delta
        self.previous = pos
        return action
