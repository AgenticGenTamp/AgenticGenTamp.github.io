// Timeline data: scenes, narration captions, grid numbers, scaling curves.
// Times are seconds on the master clock. Numbers come from the submitted paper.

// Scenes are listed with durations; start times follow from the order.
// ?protocol=stickbutton shows the StickButton version of the protocol scene instead of the Shelf one.
const PROTOCOL_ID = new URLSearchParams(location.search).get('protocol') === 'stickbutton' ? 'protocol-stick' : 'protocol';
// Surprise beats to leave out for now; list the clip file names. They stay in the markup and come
// back by removing them from this list. The remaining beats are re-timed to run back to back.
const SURPRISES_SKIP = ['clutteredstorage-main-r424-ep19', 'clutter25-main-r222-ep13', 'pushpullhook2d-windmill-source-r24-ep5'];
const SURPRISES_LENGTH = (() => {
  const sec = document.getElementById('surprises');
  let at = 0;
  for (const beat of Array.from(sec.querySelectorAll('.beat'))) {
    const src = beat.querySelector('video.clip')?.dataset.src || '';
    if (SURPRISES_SKIP.some(n => src.includes(n))) { beat.remove(); continue; }
    const from = +beat.dataset.from, dur = +beat.dataset.to - from, shift = at - from;
    beat.dataset.from = at; beat.dataset.to = at + dur;
    beat.querySelectorAll('[data-at]').forEach(el => { el.dataset.at = +el.dataset.at + shift; if (el.dataset.until) el.dataset.until = +el.dataset.until + shift; });
    at += dur;
  }
  return at;
})();

// Held-out evaluation: seconds into each mosaic at which every episode finishes (null = not solved).
const EVAL_TIMES = { protocol: [14.2, 24.4, 35.8, 46.9, 75.5, 82.5, 14.7, 24.3, 36, 45.7, 64.8, 84.1, 14.3, 23.8, 34.4, 45.1, 68, 83.8, 14.8, 25.2, 36.6, 43.9, 67.9, 93, 14.3, 24.4, 36.7, 45.6, 64.7, 83.5, 14.4, 24.2, 46.9, 44.1, 67, 91, 14.1, 25.2, 34.4, 46.2, 69.1, 91, 14.3, 24.1, 34.7, 43.8, 68.4, 83.7, 14.3, 25.8, 35.5, 46.4, 67.4, 88.2, 15.2, 24.9, 34.8, 47.6, 65.4, 91.8, 14.8, 24.4, 36, 44.3, 66.8, 91.8, 15, 25.7, 34.6, 46.8, 70.1, null, 14.7, 24, 37.2, 60.6, 64, 91.3, 14.2, 23.6, 35.3, 46.6, 64.1, 89.5, 14.3, 24.4, 34.8, 44.3, 64.1, 84.3, 14.5, 25.4, 34.3, 47.6, 64.4, 83, 14.3, 25.1, 36.7, 44.6], 'protocol-stick': [1.2, 4.7, 5.5, 7.3, 10.0, 0.6, 11.8, 5.8, 7.7, 13.9, 1.1, 11.5, 8.5, 10.8, 14.4, 3.7, 4.1, 5.6, 9.2, 11.5, 1.8, 10.0, 8.9, 9.0, 11.2, 4.6, 7.2, 5.3, 9.6, 11.1, 1.3, 8.9, 2.0, 3.4, 6.8, 5.2, 3.9, 5.6, 10.0, 11.6, 5.8, 4.7, 7.4, 6.4, 10.6, 5.5, 6.5, 7.8, 5.8, 10.9, 6.4, 1.0, 6.6, 7.1, 12.6, 3.9, 4.8, 5.9, 6.3, 13.0, 3.5, 8.6, 6.9, 11.3, 17.1, 11.0, 2.1, 5.6, 9.5, 14.4, 0.3, 10.9, 7.2, 7.4, 7.5, 6.2, 3.3, 5.6, 6.6, 11.7, 1.2, 3.8, 5.4, 8.2, 10.0, 5.1, 8.6, 6.7, 4.5, 7.6, 1.1, 4.1, 8.1, 8.3, 8.0, 7.8, 3.2, 7.4, 9.1, 15.6] };

const SCENES = (() => {
  let at = 0;
  return [
    ['hook', 13], ['problem', 36], [PROTOCOL_ID, PROTOCOL_ID === 'protocol' ? 48 : 49], ['study', 10], ['results', 18], ['source', 11],
    ['surprises', SURPRISES_LENGTH],
  ].map(([id, dur]) => { const s = { id, start: at, end: at + dur }; at += dur; return s; });
})();
const START = Object.fromEntries(SCENES.map(s => [s.id, s.start]));
START.protocol = START[PROTOCOL_ID];

// Narration draft, shown as captions. Each entry: [start, end, text].
const CAPTIONS = [
  [2, 9, 'This throw was programmed by a coding agent. Nobody gave it a planner, skills, or the simulator source.'],
  [14.8, 19, 'Task and motion planning: the same task, but every instance has different objects, positions and counts.'],
  [20.5, 26.9, 'A TAMP planner needs predicates, operators, samplers and skills, hand-designed or LLM-generated, and then searches each instance from scratch.'],
  [31, 35.89, 'Generalized TAMP writes one policy for all instances, but still on top of given abstractions.'],
  [39, 46.14, 'What if we drop the abstractions, and only let a coding agent interact with the environment and write one program?'],
  [49, 54.25, 'We give a coding agent a task description, a simulator client with reset and step, a bare sandbox and a twenty-dollar budget.'],
  [55, 61, 'It probes the black box. It writes the arm kinematics from memory and lowers the arm until it stalls on the floor: the model is off.'],
  [61, 69, 'So it holds a block, moves the arm through forty-one poses, and fits its own arm model: from thirty-nine millimetres of error to under two.'],
  [69, 73.6, 'It writes a program with a reset and a get_action method, and commits it.'],
  [74, 81.6, 'It tests on the seeds it chooses: four of five fail.'],
  [82, 88.6, 'It debugs the failing seeds and commits again, thirteen times in this run.'],
  [89, 99.27, 'Then we freeze the program and score it on one hundred held-out instances, with no language model at test time.'],
  [104.67, 113.38, 'This is one episode of seventy thousand: twenty-eight environments, five synthesis methods, five runs each, one hundred held-out instances per program.'],
  [114.71, 119, 'Twenty-eight simulated environments: kinematic and dynamic, two and three dimensions, five runs per method.'],
  [119, 120.85, 'The agent solves most of them. Fourteen environments are at ninety-five percent or above.'],
  [120.85, 122.69, 'Where a hand-engineered planner exists, the agent usually beats it, without any of its models or skills.'],
  [122.69, 125, 'LLMGenPlan, an LLM writing the program from fixed feedback, is far behind.'],
  [125, 132, 'As object counts grow, the program keeps its success rate, with about ten times less computation per instance than the planner.'],
  [132, 138.3, 'Per family, the agent is above ninety percent on the kinematic and two-dimensional tasks. Dynamic three-dimensional tasks remain hard.'],
  [140, 146.29, 'With the environment source, the agent recovers tasks the main setting cannot solve. SweepIntoDrawer goes from zero to fifty-seven percent.'],
  [146.29, 153.36, 'The price of source access: those programs call the environment planners at decision time, about seven times slower per action.'],
  [160, 167, 'The crumbs are swept with the arm instead of the tool.'],
  [169, 176, 'Failed grasps are detected and retried.'],
  [178, 185, 'Objects are pinched and carried one at a time over the divider.'],
  [187, 194, 'And a windmill-like motion with the hook.'],
];


// Problem scene instance sets: three solved episodes of one run, ordered by object count. ?env= picks one; obstruction3d is the default.
const PROBLEM_SETS = {
  clutter: { loop: 'assets/clips/grid/ClutteredRetrieval2D.mp4', clips: ['assets/clips/clutter-blackbox-r424-ep49.mp4', 'assets/clips/clutter-blackbox-r424-ep37.mp4', 'assets/clips/clutter-blackbox-r424-ep81.mp4'], caps: ['1 object', '5 objects', '15 objects'] },
  obstruction3d: { loop: 'assets/clips/grid/Obstruction3D.mp4', aspect: '4/3', clips: ['assets/clips/obstruction3d-blackbox-r24-ep26.mp4', 'assets/clips/obstruction3d-blackbox-r24-ep92.mp4', 'assets/clips/obstruction3d-blackbox-r24-ep84.mp4'], caps: ['1 obstacle', '2 obstacles', '4 obstacles'] },
};

// Lines typed on the hook: verbatim carry phase from the Tossing program of run 24 (phase chain in get_action).
const HOOK_CODE = `if ph == 'carry':
    c = cubes[self.target][0]
    if self.pt > 20 and c[2] < 0.15:
        # grasp failed -> retry
        self.grip = 0.0
        self._go('pick')
        return self._act(q, b, kv=4.0, vmax=3.2)
    if self._arm_err(q) < 0.05 or self.pt > 30:
        self.qt = qf_from_ds(self.ds_start)
        self._go('windup')
    return self._act(q, b, kv=8.0, vmax=4.5)

if ph == 'windup':
`;

// Terminal script for the protocol scene: verbatim commands, code, outputs and commit messages
// from the StickButton main-setting run 42 (times are seconds into the scene).
const TERM = [
 {
  "at": 6.0,
  "type": "type",
  "cps": 200,
  "text": "from env_client import make_env\nenv = make_env()\nfor seed in range(6):\n    obs, info = env.reset(seed=seed)\n    for name in sorted(obs.get_object_names()):\n        o = obs.get_object_from_name(name)\n        print(name, o.type.name, {f: round(float(obs.get(o, f)), 4) for f in obs.type_features[o.type]})"
 },
 {
  "at": 7.84,
  "type": "print",
  "text": "=== seed 0 info {'object_count': 3}"
 },
 {
  "at": 7.96,
  "type": "print",
  "text": "  button0 circle {'x': 2.1126, 'y': 1.8008, 'theta': 0.0, 'radius': 0.05}"
 },
 {
  "at": 8.08,
  "type": "print",
  "text": "  button1 circle {'x': 1.8983, 'y': 2.2942, 'theta': 0.0, 'radius': 0.05}"
 },
 {
  "at": 8.2,
  "type": "print",
  "text": "  button2 circle {'x': 2.8239, 'y': 0.0566, 'theta': 0.0, 'radius': 0.05}"
 },
 {
  "at": 8.32,
  "type": "print",
  "text": "  robot crv_robot {'x': 2.1472, 'y': 0.4754, 'theta': -2.8841, 'base_radius': 0.1, 'arm_joint': "
 },
 {
  "at": 8.44,
  "type": "print",
  "text": "  stick rectangle {'x': 0.057, 'y': 1.0316, 'theta': 0.0, 'width': 0.05, 'height': 1.25}"
 },
 {
  "at": 8.72,
  "type": "type",
  "cps": 200,
  "text": "\n# push the base into each wall until it stops moving\nfor vec in ([0.05,0,0,0,0], [0,0.05,0,0,0], [-0.05,0,0,0,0], [0,-0.05,0,0,0]):\n    prev = None\n    while True:\n        obs,rew,term,trunc,info = env.step(np.array(vec, dtype=np.float32))\n        cur = (round(obs.get(robot,\"x\"),3), round(obs.get(robot,\"y\"),3))\n        if cur == prev: break\n        prev = cur\n    print(\"stopped at\", cur)"
 },
 {
  "at": 11.44,
  "type": "print",
  "text": "stopped at (3.335, 0.948)"
 },
 {
  "at": 11.76,
  "type": "print",
  "text": "stopped at (3.335, 1.148)"
 },
 {
  "at": 12.72,
  "type": "print",
  "text": "stopped at (0.135, 1.148)"
 },
 {
  "at": 13.36,
  "type": "print",
  "text": "stopped at (0.135, 0.148)"
 },
 {
  "at": 14.0,
  "type": "clear"
 },
 {
  "at": 14.0,
  "type": "type",
  "cps": 200,
  "text": "class GeneratedApproach:\n    def reset(self, state, info):\n        ...\n\n    def get_action(self, state):\n        try:\n            a = self._get_action(state)\n            a = np.asarray(a, dtype=np.float32).reshape(5)\n            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)\n            lo = np.asarray(self.action_space.low, dtype=np.float32)\n            hi = np.asarray(self.action_space.high, dtype=np.float32)\n            return np.clip(a, lo, hi).astype(np.float32)\n        except Exception:\n            v = 1.0 if getattr(self, \"carrying\", False) else 0.0\n            return np.array([0.0, 0.0, 0.0, 0.0, v], dtype=np.float32)"
 },
 {
  "at": 17.29,
  "type": "commit",
  "text": "17c7aa1  v1: base tour + grasp-from-below stick plan"
 },
 {
  "at": 19.0,
  "type": "clear"
 },
 {
  "at": 19.0,
  "type": "cmd",
  "text": "$ python test_approach.py 0 1 2 3 4 5 6 7 8 9"
 },
 {
  "at": 19.64,
  "type": "print",
  "text": "seed 0: steps=1000 term=False buttons=1/3"
 },
 {
  "at": 19.8,
  "type": "print",
  "text": "seed 1: steps=30 term=True buttons=2/2"
 },
 {
  "at": 19.96,
  "type": "print",
  "text": "seed 2: steps=1000 term=False buttons=2/3"
 },
 {
  "at": 20.12,
  "type": "print",
  "text": "seed 3: steps=1000 term=False buttons=2/3"
 },
 {
  "at": 20.28,
  "type": "print",
  "text": "seed 4: steps=1000 term=False buttons=2/3"
 },
 {
  "at": 20.44,
  "type": "print",
  "text": "seed 8: steps=21 term=True buttons=3/3"
 },
 {
  "at": 20.6,
  "type": "print",
  "text": "mean steps 706.5 fails [0, 2, 3, 4, 5, 7, 9]"
 },
 {
  "at": 21.4,
  "type": "type",
  "cps": 200,
  "text": "\n# sweep.py: 300 seeds in parallel, keep the worst\nfor s in range(lo,hi):\n    obs,info=env.reset(seed=s)\n    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)\n    for i in range(env.max_steps):\n        obs,r,term,trunc,inf=env.step(ap.get_action(obs))\n        if term or trunc: break\nfails=[s for s,st,t in res if not t]\nworst=sorted([(st,s) for s,st,t in res if t])[-10:]"
 },
 {
  "at": 24.2,
  "type": "print",
  "text": "n 300 solved 298 mean steps 55.8 max 139 fails [455, 496]"
 },
 {
  "at": 24.44,
  "type": "print",
  "text": "worst: [(114, 479), (114, 563), (116, 518), (116, 542), (116, 611), (117, 475), (117, 628), (118, 620), (122, 499), (139, 677)]"
 },
 {
  "at": 27.0,
  "type": "clear"
 },
 {
  "at": 27.0,
  "type": "cmd",
  "text": "$ python dbg.py 455"
 },
 {
  "at": 27.47,
  "type": "print",
  "text": "button2 1.754 1.155"
 },
 {
  "at": 27.54,
  "type": "print",
  "text": "stick 3.446 0.978"
 },
 {
  "at": 27.7,
  "type": "print",
  "text": "40 act [0. 0. 0. 0. 0.] rob 3.233 0.953 0.0 0.1 stick 3.446 0.978 carry False stall 1 task ('grasp', None)"
 },
 {
  "at": 27.86,
  "type": "print",
  "text": "45 act [0. -0.05 0. -0.1 0.] rob 3.233 0.903 0.0 0.1 stick 3.446 0.978 carry False stall 7 task ('grasp', None)"
 },
 {
  "at": 28.01,
  "type": "print",
  "text": "60 act [0. -0.05 0. -0.1 0.] rob 3.233 0.903 0.0 0.1 stick 3.446 0.978 carry False stall 7 task ('grasp', None)"
 },
 {
  "at": 28.94,
  "type": "commit",
  "text": "972f301  robust grasp: staging+creep with vacuum, proper stall detection"
 },
 {
  "at": 29.72,
  "type": "print",
  "text": "n 400 solved 400 mean steps 53.8 max 128 fails []"
 },
 {
  "at": 30.5,
  "type": "commit",
  "text": "fba7659  lookahead-aware press-point selection (base and stick)"
 },
 {
  "at": 30.73,
  "type": "commit",
  "text": "8a744e8  reposition (drop+regrasp) fallback, rotate-in-place stall recovery"
 },
 {
  "at": 30.97,
  "type": "commit",
  "text": "15980c5  add lateral grasp fallback for wall-hugging sticks + carry-aware x limits"
 },
 {
  "at": 31.2,
  "type": "commit",
  "text": "16f7419  better stick routing (diagonal, tighter pass line), pre-rotate toward grasp"
 },
 {
  "at": 31.43,
  "type": "commit",
  "text": "2b1543b  wall-aware gripper parking orientation; better stall recovery ordering"
 },
 {
  "at": 31.67,
  "type": "commit",
  "text": "ab7694f  robustness pass: exception guard, state init, watchdog fixes"
 },
 {
  "at": 31.9,
  "type": "commit",
  "text": "c373fa9  clean up exploration scripts; validated 300/300 unseen seeds, mean 52 steps"
 }
];

// Terminal script for the Shelf version of the protocol scene: verbatim from the Shelf main-setting run 222.
const TERM_SHELF = [
 {
  "at": 6,
  "type": "type",
  "cps": 200,
  "text": "from env_client import make_env\nenv = make_env()\nobs, info = env.reset(seed=1)\nfor name in sorted(obs.get_object_names()):\n    o = obs.get_object_from_name(name)\n    print(name, o.type.name, {f: round(float(obs.get(o, f)), 4) for f in obs.type_features[o.type]})"
 },
 {
  "at": 7.76,
  "type": "print",
  "text": "info: {'object_count': 2}"
 },
 {
  "at": 7.88,
  "type": "print",
  "text": "cube1 mujoco_movable_object {'x': 0.4627, 'y': -1.0087, 'z': 0.0199, 'qw': 1.0, ..., 'bb_x': 0.04}"
 },
 {
  "at": 8,
  "type": "print",
  "text": "cupboard_1 mujoco_fixture {'x': 1.5, 'y': 0.0, 'z': 0.0041, 'qw': 0.7071, ...}"
 },
 {
  "at": 8.12,
  "type": "print",
  "text": "robot mujoco_tidybot_robot {'pos_base_x', 'pos_base_y', 'pos_base_rot', 'pos_arm_joint1' ... 'pos_arm_joint7', 'pos_gripper'}"
 },
 {
  "at": 8.4,
  "type": "type",
  "cps": 200,
  "text": "from fkfast import ik, fk_all   # Kinova Gen3 kinematics, written from memory\n# probe_floor.py: lower the arm 3 cm at a time until it stalls on the floor\nfor k in range(30):\n    zw -= 0.03\n    qt, ike = ik(np.array([r, 0, zw]), Rup, qprev, tool=L)\n    o, st = drive(e, o, qt, maxsteps=60)\n    print('zw', zw, 'steps', st, 'err', maxabs(qt - getq(o)), 'modelz', fk_all(getq(o), L)[-1][2, 3])"
 },
 {
  "at": 10.6,
  "type": "print",
  "text": "zw -0.3 steps 5 err 0.0027 modelz -0.2999"
 },
 {
  "at": 10.8,
  "type": "print",
  "text": "zw -0.33 steps 7 err 0.0025 modelz -0.3301"
 },
 {
  "at": 11,
  "type": "print",
  "text": "zw -0.36 steps 6 err 0.0030 modelz -0.3598"
 },
 {
  "at": 11.4,
  "type": "print",
  "text": "zw -0.39 steps 60 err 0.0447 modelz -0.3706"
 },
 {
  "at": 11.55,
  "type": "print",
  "text": "STALL"
 },
 {
  "at": 11.75,
  "type": "print",
  "text": "# on the floor, yet the model puts the fingertip 2.4 cm above it: mount and gripper offsets are wrong"
 },
 {
  "at": 14.72,
  "type": "type",
  "cps": 200,
  "text": "\n# calib.py: hold a cube, move the arm through random poses, record joint angles and cube position\nfor k in range(14):\n    tx=rng.uniform(0.25,0.55); ty=rng.uniform(-0.3,0.3); tz=rng.uniform(0.25,0.6)-Z0\n    qn,ike=ik(np.array([tx,ty,tz]),Rt,qcur,tool=0.20)\n    o,st=drive(e,o,qn,grip=1.0,maxsteps=200,base=base,tol=0.004)\n    data.append({\"q\":getq(o).tolist(),\"base\":getb(o).tolist(),\"cube\":cubes(o)[name].tolist()})"
 },
 {
  "at": 17.12,
  "type": "print",
  "text": "collected 14 | collected 14 | collected 13"
 },
 {
  "at": 17.6,
  "type": "type",
  "cps": 200,
  "text": "\n# fit.py: least squares over the arm mount and tool offsets\nr=least_squares(resid,np.array([0.148,0.01,0.415,0,0,-0.2]))"
 },
 {
  "at": 18.72,
  "type": "print",
  "text": "N 41"
 },
 {
  "at": 18.88,
  "type": "print",
  "text": "mount [1.201e-01 1.000e-04 3.947e-01] delta [-2.000e-04  1.110e-02 -2.057e-01]"
 },
 {
  "at": 19.04,
  "type": "print",
  "text": "rms [0.0006 0.0012 0.0012] max [0.0019 0.0035 0.0026]"
 },
 {
  "at": 20,
  "type": "clear"
 },
 {
  "at": 20,
  "type": "type",
  "cps": 200,
  "text": "def get_action(self, state):\n    q, b, g, cubes = self._read(state)\n    name = self.order[self.idx]\n    cp = cubes.get(name)\n    ph = self.phase\n    # Phase 0: go to hover above cube\n    if ph == 0:\n        bt = (cp[0] - self.GRAB_R, cp[1], 0.0)\n        a = self._act(q, self.q_hover, b, bt, 0.0)\n        if qerr(self.q_hover) < 0.02 and berr(bt) < 0.008:\n            self.phase = 1\n        return a\n    # Phase 1: descend\n    if ph == 1:\n        bt = (cp[0] - self.GRAB_R, cp[1], 0.0)\n        a = self._act(q, self.q_grasp, b, bt, 0.0)\n        if qerr(self.q_grasp) < 0.01:\n            self.phase = 2\n        return a\n    # Phase 2: close gripper"
 },
 {
  "at": 23.29,
  "type": "commit",
  "text": "ade02b3  first end-to-end approach: grasp cubes from floor, insert on top shelf of cupboard"
 },
 {
  "at": 25,
  "type": "clear"
 },
 {
  "at": 25,
  "type": "cmd",
  "text": "$ for s in 0 2 3 4 5; do python test_approach.py $s; done"
 },
 {
  "at": 25.64,
  "type": "print",
  "text": "seed 0 n=2 steps 672 term True"
 },
 {
  "at": 25.8,
  "type": "print",
  "text": "seed 2 n=2 steps 1000 term False"
 },
 {
  "at": 25.96,
  "type": "print",
  "text": "seed 3 n=2 steps 1000 term False"
 },
 {
  "at": 26.12,
  "type": "print",
  "text": "seed 4 n=2 steps 1000 term False"
 },
 {
  "at": 26.28,
  "type": "print",
  "text": "seed 5 n=2 steps 1000 term False"
 },
 {
  "at": 27.4,
  "type": "cmd",
  "text": "$ for s in 2 3 4 5; do python test_approach.py 8 $s; done   # eight cubes"
 },
 {
  "at": 28.36,
  "type": "print",
  "text": "seed 3 n=8 steps 1000 TERM False   cube3 [-0.014, -0.671, 0.02] still on the floor"
 },
 {
  "at": 28.52,
  "type": "print",
  "text": "seed 2 n=8 steps 1000 TERM False   cube8 [0.604, ...] still on the floor"
 },
 {
  "at": 33,
  "type": "clear"
 },
 {
  "at": 33,
  "type": "commit",
  "text": "bdb2e26  speed up: merge phases, shorter gripper waits, lower hover, overlap retract with arm return"
 },
 {
  "at": 33.78,
  "type": "commit",
  "text": "b8775bf  robust cycle: retarget unplaced cubes, verify grasp, monotonic slot fill"
 },
 {
  "at": 34.56,
  "type": "cmd",
  "text": "$ for s in 1 2 4 6 8 9 10 11; do python test_approach.py $s; done"
 },
 {
  "at": 35.18,
  "type": "print",
  "text": "seed 8 n=2 steps 409 TERM True"
 },
 {
  "at": 35.33,
  "type": "print",
  "text": "seed 4 n=2 steps 436 TERM True"
 },
 {
  "at": 35.49,
  "type": "print",
  "text": "seed 10 n=2 steps 434 TERM True"
 },
 {
  "at": 35.64,
  "type": "print",
  "text": "seed 2 n=2 steps 432 TERM True"
 },
 {
  "at": 36.5,
  "type": "commit",
  "text": "a5850b1  adopt optimized grasp/carry config pair: 8 cubes in ~830 steps"
 },
 {
  "at": 36.73,
  "type": "commit",
  "text": "0a0b459  add skip list for repeatedly failed grasps"
 },
 {
  "at": 36.97,
  "type": "commit",
  "text": "4c20f98  robust cupboard lookup"
 },
 {
  "at": 37.2,
  "type": "commit",
  "text": "f66751b  narrow shelf slot span to avoid side-wall collisions with 8 cubes"
 }
];

// Extra cue times for animations that are driven in code rather than by reveal elements.
const EXTRA_CUES = [
  // Problem scene: moments driven in code (instances to the side, planners, outputs, fans).
  ...[6.5, 11, 12.3, 20.5, 22.3, 30.2, 31].map(x => START.problem + x),
  // StickButton protocol: every block the terminal starts typing, and every commit line. The Shelf
  // protocol paces itself by its headlines alone, one cue per step.
  ...(PROTOCOL_ID === 'protocol' ? [] : TERM.filter((ev, i, arr) => (ev.type === 'type' || ev.type === 'cmd' || (ev.type === 'commit' && arr[i - 1].type !== 'commit'))).map(ev => START.protocol + ev.at)),
  START.results + 3, START.results + 7, START.results + 12,
];

// Environment grid, success rates from the paper's Tables I and II: agent = AgenticGenPlan (Opus 5, main setting),
// planner = TAMP planner (null where the benchmark supplies none), genplan = LLMGenPlan (Opus 5).
const GRID = [
  { fam: 'Kinematic 2D', name: 'StickButton', img: 'StickButton2D', agent: 1.00, planner: 0.36, genplan: 0.83 },
  { name: 'Obstruction', img: 'Obstruction2D', agent: 1.00, planner: 0.41, genplan: 0.82 },
  { name: 'ClutteredStorage', img: 'ClutteredStorage2D', agent: 0.99, planner: 0.19, genplan: 0.02 },
  { name: 'ClutteredRetrieval', img: 'ClutteredRetrieval2D', agent: 0.81, planner: 0.51, genplan: 0.32 },
  { name: 'Motion', img: 'Motion2D', agent: 1.00, planner: 0.71, genplan: 0.97 },
  { name: 'PushPullHook', img: 'PushPullHook2D', agent: 0.98, planner: null, genplan: 0.12 },
  { fam: 'Dynamic 2D', name: 'Obstruction', img: 'DynObstruction2D', agent: 0.88, planner: 0.24, genplan: 0.49 },
  { name: 'PushPullHook', img: 'DynPushPullHook2D', agent: 0.83, planner: 0.13, genplan: 0.00 },
  { name: 'PushT', img: 'DynPushT2D', agent: 1.00, planner: null, genplan: 0.17 },
  { name: 'ScoopPour', img: 'DynScoopPour2D', agent: 0.97, planner: null, genplan: 0.00 },
  { fam: 'Kinematic 3D', name: 'Obstruction', img: 'Obstruction3D', agent: 0.85, planner: null, genplan: 0.00 },
  { name: 'Packing', img: 'Packing3D', agent: 0.66, planner: 0.67, genplan: 0.14 },
  { name: 'Transport', img: 'Transport3D', agent: 0.99, planner: 0.63, genplan: 0.05 },
  { name: 'Table', img: 'Table3D', agent: 1.00, planner: null, genplan: 0.85 },
  { name: 'BaseMotion', img: 'BaseMotion3D', agent: 1.00, planner: 1.00, genplan: 1.00 },
  { fam: 'Dynamic 3D', name: 'BalanceBeam', img: 'BalanceBeam3D', agent: 0.97, planner: null, genplan: 0.00 },
  { name: 'ConstrainedCupboard', img: 'ConstrainedCupboard3D', agent: 0.08, planner: null, genplan: 0.00 },
  { name: 'Dynamo', img: 'Dynamo3D', agent: 0.92, planner: null, genplan: 0.98 },
  { name: 'Rearrange', img: 'Rearrange3D', agent: 0.58, planner: null, genplan: 0.00 },
  { name: 'ScoopPour', img: 'ScoopPour3D', agent: 0.09, planner: null, genplan: 0.00 },
  { name: 'Shelf', img: 'Shelf3D', agent: 0.60, planner: 0.27, genplan: 0.00 },
  { name: 'SortClutteredBlocks', img: 'SortClutteredBlocks3D', agent: 0.30, planner: null, genplan: 0.00 },
  { name: 'SweepIntoDrawer', img: 'SweepIntoDrawer3D', agent: 0.00, planner: 0.00, genplan: 0.00 },
  { name: 'SweepSimple', img: 'SweepSimple3D', agent: 0.00, planner: null, genplan: 0.00 },
  { name: 'Tossing', img: 'Tossing3D', agent: 0.99, planner: 0.77, genplan: 0.00 },
  { fam: 'PDDLStream', name: 'Packing', img: 'PddlPacking', agent: 0.99, planner: 0.54, genplan: 0.21 },
  { name: 'Blocked', img: 'PddlBlocked', agent: 0.38, planner: 0.74, genplan: 0.00 },
  { name: 'Rovers', img: 'PddlRovers', agent: 0.98, planner: 0.30, genplan: 0.86 },
];

// Table III: policy-computation time per action (ms) on 44 matched seeds across 13 environments
// where both settings reach 100% held-out success; equal-environment means with [min, max].
const EFFICIENCY = [
  { name: 'AgenticGenPlan', ms: 4.0, range: [0.04, 29.2] },
  { name: 'AgenticGenPlan + source', ms: 27.9, range: [0.07, 96.6] },
];

// Aggregate scaling over the 15 planner environments with variable object counts (Figure 1 data).
const SCALING = {
  levels: [1, 2, 3, 4, 5],
  program: { success: [0.887, 0.912, 0.868, 0.777, 0.748], time: [1.10, 1.52, 2.02, 3.57, 4.75] },
  planner: { success: [0.791, 0.653, 0.395, 0.236, 0.170], time: [8.89, 18.99, 32.68, 40.10, 42.99] },
  // LLMGenPlan from the teaser's plotted-aggregate (14 environments; runtime over the 4 timed ones).
  genplan: { success: [0.369, 0.313, 0.269, 0.233, 0.201], time: [0.80, 1.08, 1.92, 2.63, 2.88] },
};
