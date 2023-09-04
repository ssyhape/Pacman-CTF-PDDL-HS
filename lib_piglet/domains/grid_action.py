# domains/grid_action.py
# 
# describes a valid action in a grid domain and specifies its cost
# 
# @author: dharabor
# @created: 2020-07-15

import sys
from enum import IntEnum

class Move_Actions(IntEnum):
    MOVE_UP = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_DOWN = 3
    MOVE_WAIT = 4
    MOVE_UP_LEFT = 5
    MOVE_UP_RIGHT = 6
    MOVE_DOWN_RIGHT = 7
    MOVE_DOWN_LEFT = 8

class grid_action:

    def __init__(self):
        self.move_ = Move_Actions.MOVE_WAIT
        self.cost_ = 1

    def print(self):
        if(self.move_ == Move_Actions.MOVE_UP):
            print("UP " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_DOWN):
            print("DOWN " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_LEFT):
            print("LEFT " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_RIGHT):
            print("RIGHT " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_UP_LEFT):
            print("UP LEFT " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_UP_RIGHT):
            print("UP RIGHT " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_DOWN_RIGHT):
            print("DOWN RIGHT " + str(self.cost_))
        elif(self.move_ == Move_Actions.MOVE_DOWN_LEFT):
            print("DOWN LEFT " + str(self.cost_))
        else:
            print("WAIT " + str(self.cost_))

