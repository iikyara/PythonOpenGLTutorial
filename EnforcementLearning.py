class Environment:
    def __init__(self, name="Environment"):
        self.name = name
        self.enable_actions = []
        self.reward = 0
        self.terminal = False

    def update(self, action):
        pass

    def execute_action(self, action):
        self.update(action)

    def getReward(self):
        return self.reward

    def draw(self):
        pass

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def reset(self):
        pass

class Agent:
    def __init__(self, name="Agent"):
        self.name = name

    def update(self):
        pass


    def reset(self):
        pass
