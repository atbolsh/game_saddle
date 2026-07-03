from math import pi

class Settings:
    def __init__(self, gameSize=64, direction=0, agent_x=0.5, agent_y=0.5, agent_r=0.05, gold_r=0.01, gold=None, walls=None):#, indicator_length=None):
#        if indicator_length is None:
#            indicator_length = agent_r
        if gold is None:
            gold = []
        if walls is None:
            walls = []
        self.gameSize = gameSize
#        self.indicator_length = indicator_length
        self.direction = direction
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.agent_r = agent_r
        self.gold_r = gold_r
        self.gold = gold
        self.walls = walls



default2_5 = Settings(800)

tool_use_advanced_2_5 = \
    Settings(64, \
#             indicator_length = 400/800.0,
             agent_x = 600.0/800,
             agent_y = 600.0/800,
             agent_r = 40.0/800,
             gold_r = 1.0/64,
             gold = [[170/800.0, 269/800.0], [400/800.0, 400/800.0]],
             walls = [
                         [0, 0, 50/800.0, 800/800.0, 0],
                         [0, 0, 800/800.0, 50/800.0, 0],
                         [0, (800 - 50)/800.0, 800/800.0, 50/800.0, 0],
                         [(800 - 50)/800.0, 0, 50/800.0, 800/800.0, 0],
                         [100/800.0, 100/800.0, 50/800.0, 450/800.0, 0],
                         [100/800.0, 100/800.0, 50/800.0, 450/800.0, -pi/6]
                     ]
            )

BIG_tool_use_advanced_2_5 = \
    Settings(800, \
             agent_x = 600.0/800,
             agent_y = 600.0/800,
             agent_r = 40.0/800,
             gold_r = 1.0/64,
             gold = [[170/800.0, 269/800.0], [400/800.0, 400/800.0]],
             walls = [
                         [0, 0, 50/800.0, 800/800.0, 0],
                         [0, 0, 800/800.0, 50/800.0, 0],
                         [0, (800 - 50)/800.0, 800/800.0, 50/800.0, 0],
                         [(800 - 50)/800.0, 0, 50/800.0, 800/800.0, 0],
                         [100/800.0, 100/800.0, 50/800.0, 450/800.0, 0],
                         [100/800.0, 100/800.0, 50/800.0, 450/800.0, -pi/6]
                     ]
            )
