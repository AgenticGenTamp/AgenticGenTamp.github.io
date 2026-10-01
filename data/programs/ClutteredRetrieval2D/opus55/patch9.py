src = open('approach.py').read()
src = src.replace('''    def get_action(self, state):
        self._parse(state)''', '''    def get_action(self, state):
        try:
            return self._get_action(state)
        except Exception:
            if DEBUG:
                raise
            self.queue = []
            self.pending = None
            self.cached_carry = None
            self.expected = None
            return self._act(np.zeros(4), 1.0 if self.held else 0.0)

    def _get_action(self, state):
        self._parse(state)''')
open('approach.py','w').write(src)
