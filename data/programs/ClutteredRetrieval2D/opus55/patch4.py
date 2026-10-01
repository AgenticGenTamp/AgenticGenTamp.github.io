src = open('approach.py').read()
src = src.replace('''        cands = self.grasp_candidates('target_block')
        good, pblock = self.place_blockers(cands)
        if good:''', '''        prio = [n for n in self.priority_remove if n in self.rects]
        self.priority_remove = []
        for name in prio:
            path = self.plan_grasp(name, time_budget=1.0)
            if path is not None:
                self._go_grasp(path, name)
                return
        cands = self.grasp_candidates('target_block')
        good, pblock = self.place_blockers(cands)
        if good:''')
src = src.replace('''        order = [n for n in self.priority_remove if n in self.rects]
        self.priority_remove = []
        if not good''', '''        order = []
        if not good''')
src = src.replace("                self.priority_remove = self.escape_blockers()\n", "                self.priority_remove = self.escape_blockers()\n                if DEBUG: print('escape blockers', self.priority_remove)\n")
src = src.replace("MARGIN = 0.004\n", "MARGIN = 0.004\nDEBUG = False\n")
open('approach.py','w').write(src)
