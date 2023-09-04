
from lib_piglet.expanders.base_expander import base_expander
from lib_piglet.domains.n_puzzle import n_puzzle, puzzle_state, Puzzle_Actions
from lib_piglet.search.search_node import search_node
from enum import IntEnum



class puzzle_action:
    
    def __init__(self, action: int, cost:int):
        self.move_: int = action
        self.cost_: int = cost
        self.next_x_index: int = 0
        
class n_puzzle_expander(base_expander):


    def __init__(self,puzzle: n_puzzle):
        self.domain_: n_puzzle = puzzle
        self.succ_: list = []
        self.swap_offset_: list = None
        self.__init_swap_offset_()

    def expand(self, current_node: search_node):
        self.succ_.clear()
        current_state: puzzle_state = current_node.state_
        for valid_action in self.get_actions(current_node):

            # generate new state based on valid action.
            successor = self.__move(current_state,valid_action)
            self.succ_.append((successor, valid_action))
        return self.succ_[:]

    def get_actions(self,current: search_node):
        retval = []
        for action in range(0, len(self.swap_offset_)):
            new_x_index = current.state_.x_index_ + self.swap_offset_[action]
            if not self.is_valid_move(current.state_.x_index_, new_x_index):
                continue
            valid_action = puzzle_action(action,1)
            valid_action.next_x_index = new_x_index
            retval.append(valid_action)
        return retval

    
    # Return is the new index/location valid
    # @param old_index The old index for x(white space)
    # @param new_index The new index for x(white space)
    # @return bool True if new_index is valid
    def is_valid_move(self,old_index:int, new_index:int):
        if new_index < 0 or new_index >= self.domain_.size_:
            return False
        curr_x = old_index // self.domain_.width_
        curr_y = old_index % self.domain_.width_
        next_x = new_index // self.domain_.width_
        next_y = new_index % self.domain_.width_
        return abs(next_x - curr_x) + abs(next_y - curr_y) < 2

    def __move(self, current: puzzle_state, valid_action: puzzle_action ):
        new_x_index = valid_action.next_x_index
        action = valid_action.move_
        new_list = current.state_list_[:]
        temp = new_list[current.x_index_]
        new_list[current.x_index_] = new_list[new_x_index]
        new_list[new_x_index] = temp
        return puzzle_state(new_list,new_x_index,action)
    
    def __init_swap_offset_(self):
        self.swap_offset_ = [None]*4
        self.swap_offset_[Puzzle_Actions.SWAP_UP] = -1*self.domain_.width_
        self.swap_offset_[Puzzle_Actions.SWAP_DOWN] = self.domain_.width_
        self.swap_offset_[Puzzle_Actions.SWAP_LEFT] = -1
        self.swap_offset_[Puzzle_Actions.SWAP_RIGHT] = 1

    def __str__(self):
        return str(self.domain_)









