src = open('approach.py').read()
old = '''            path = self.plan_dump()
            if path is not None:
                self.enqueue_path(path, 1.0)
            self._release()
            return'''
assert old in src
src = src.replace(old, '''            path = self.plan_dump()
            if path is not None:
                self.enqueue_path(path, 1.0)
            else:
                self.fail_count[self.held] = self.fail_count.get(self.held, 0) + 1
                self.priority_remove = self.escape_blockers()
                if DEBUG: print('dump failed; escape blockers', self.priority_remove)
            self._release()
            return''')
src = src.replace('''        order = [onames[i] for i in np.argsort(-cnt) if cnt[i] > 0 and onames[i] != 'target_block']''',
 '''        order = [onames[i] for i in np.argsort(-cnt) if cnt[i] > 0 and onames[i] != 'target_block'
                 and self.fail_count.get(onames[i], 0) < 2]''')
src = src.replace('''        for name in order[:6]:''', '''        order = [n for n in order if self.fail_count.get(n, 0) < 2] + \\
            [n for n in order if self.fail_count.get(n, 0) >= 2]
        for name in order[:6]:''')
open('approach.py','w').write(src)
