import random

class RandomBot:

    def select_action(self, env):
        return random.choice(env.legal_actions())