from approach import GeneratedApproach as Base
class GeneratedApproach(Base):
    def _action(self,state):
        if self.count>=350:self.angular_weight=.8
        return super()._action(state)
