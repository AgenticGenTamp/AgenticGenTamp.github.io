import numpy as np
from prototype_agent import GeneratedApproach as Base


class GeneratedApproach(Base):
    def reset(self, state, info):
        super().reset(state, info)
        s = np.asarray(state)
        g = s[29:31] - s[20:22]
        g /= max(float(np.linalg.norm(g)), 1e-6)
        self.child_continuous = s[21] > 1.72 and g[1] < -.65
        # Parent currently contains the competing always-on experiment; disable
        # it so this wrapper isolates the gated variant.
        self.high_down = False

    def get_action(self, state):
        s = np.asarray(state)
        # Once the button has cleared the divider boundary, return to the normal
        # recontact loop; otherwise a slipped hook keeps translating forever.
        continuous = self.child_continuous and self.phase == 8 and s[21] > 1.72
        if continuous:
            self.age = 0
            self.push_start = s[20:22].copy()
        a = super().get_action(s)
        if continuous and self.phase == 8:
            goal = s[29:31] - s[20:22]
            a[:2] = .006 * goal / max(float(np.linalg.norm(goal)), 1e-6)
        return a
