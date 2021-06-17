from EnforcementLearning import Environment, Agent

class STMEnvironment(Environment):
    def __init__(self, agents=None, agent_num=10):
        super(STMEnvironment, self).__init__("SketchToEnvironment")
        self.agents = agents or [Agent() for i in range(agent_num)]

    def update(self):
        for agent in self.agents:
            agent.update()
