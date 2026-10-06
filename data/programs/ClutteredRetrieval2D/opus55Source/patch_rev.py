s=open('approach.py').read()
old="""                    self.plan = self._mk_plan(path, 1.0, 0.0)
                    return True
            if path is None:
                if DEBUG:
                    print('DBG held-fail', held"""
new="""                    self.plan = self._mk_plan(path, 1.0, 0.0)
                    self.last_drop_path = [np.array(p, float) for p in path]
                    return True
            if path is None:
                if DEBUG:
                    print('DBG held-fail', held"""
assert old in s; s=s.replace(old,new)
old="""                self.plan = [(q0, 0.0)]
                return True
            self.plan = self._mk_plan(path, 1.0, 0.0)
            return True
        chk0 = Checker(model, self._obstacles(rects, set()), [])"""
new="""                self.plan = [(q0, 0.0)]
                return True
            self.plan = self._mk_plan(path, 1.0, 0.0)
            self.last_drop_path = [np.array(p, float) for p in path]
            return True
        chk0 = Checker(model, self._obstacles(rects, set()), [])
        ok = self._make_plan_free(r, rects, model, q0, chk0, tgt, reg, budget)
        if not ok:
            ok = self._retrace_drop(chk0, q0)
        else:
            self.last_drop_path = None
        return ok

    def _retrace_drop(self, chk0, q0):
        \"\"\"Back out along the path used for the last drop (escape from pockets).\"\"\"
        p = self.last_drop_path
        self.last_drop_path = None
        if not p or np.abs(cfg_diff(p[-1], q0)).max() > 1e-3:
            return False
        rev = p[::-1][1:]
        out = []
        for q in rev:
            if not chk0.free(q, margin=0.0):
                break
            out.append(q)
        if len(out) < 2:
            return False
        if DEBUG:
            print('DBG retrace', len(out))
        self.plan = self._mk_plan(out, 0.0, 0.0)
        return True

    def _make_plan_free(self, r, rects, model, q0, chk0, tgt, reg, budget):"""
assert old in s; s=s.replace(old,new)
s=s.replace("        self.path_blockers = []\n","        self.path_blockers = []\n        self.last_drop_path = None\n",1)
open('approach.py','w').write(s)
