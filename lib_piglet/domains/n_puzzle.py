# domains/n_puzzle_h.py
# This module implements a n_puzzle domain
#
# For n_puzzle, a state is a puzzle_state object. __eq__ is defined in puzzle_state for equal check.
# @author: mike
# @created: 2020-07-16

import math,sys
from enum import IntEnum

class Puzzle_Actions(IntEnum):
    SWAP_UP = 0
    SWAP_LEFT = 1
    SWAP_RIGHT = 2
    SWAP_DOWN = 3
    START = -1
    GOAL = -2

class puzzle_state:

    # Print current state in a nice layout
    def print(self):
        width = math.sqrt(len(self.state_list_))
        str_ = "\n"
        for i in range(1, len(self.state_list_) + 1):
            str_ += str(self.state_list_[i - 1])
            if i % width == 0:
                str_ += "\n"
            else:
                str_ += "\t"
        print(str_)

    def __init__(self, alist: list, x_index: int, from_action: int = Puzzle_Actions.START):
        self.state_list_: list = alist
        self.from_action_: int  = from_action
        self.x_index_: int = x_index

    def __eq__(self, other):
        if type(other) == str:
            return str(self.state_list_) == other
        if type(other) == list:
            return self.state_list_ == other
        return self.state_list_ == other.state_list_

    def __str__(self):
        return Puzzle_Actions(self.from_action_).name

    def __repr__(self):
        return Puzzle_Actions(self.from_action_).name

    def __hash__(self):
        return hash(str(self.state_list_))


class n_puzzle:

    # Initialize a problem
    # @param width The width of the puzzle
    def __init__(self, width: int):
        self.width_: int = width
        self.size_: int = width*width
        goal_list = ["x"] + list(range(1, self.width_*self.width_))
        self.goal_: puzzle_state = puzzle_state(goal_list, self.width_-1, -2)
        self.start_: puzzle_state = None
        self.domain_file_:str = width
    
    def is_goal(self, current_state, goal_state):
        return current_state == goal_state


    def set_start(self, alist: list):
        puzzle_list = []
        if len(alist) != self.size_:
            print("err; The length of puzzle not equal to puzzle width^2", file = sys.stderr)
            exit(1)
        for item in alist:
            if type(item) == str:
                if item.isnumeric():
                    num = int(item)
                else:
                    num = "x"
            else:
                try:
                    num = int(item)
                except:
                    print("err; unknown element type for: {item}".format(item), file=sys.stderr)
                    exit(1)
            if num == 0:
                num = "x"
            if num!="x" and (num <= 0 or num >= self.size_):
                print("err; Number {} not in range 1~{}".format(num, self.size_ - 1), file=sys.stderr)
                exit(1)


            if num in puzzle_list:
                print("You can't have two {} in one puzzle".format(num), file = sys.stderr)
                exit(1)
            puzzle_list.append(num)

        self.start_ = puzzle_state(puzzle_list,puzzle_list.index("x"))
        # if not self.is_solvable():
        #     print("The given puzzle {} is not solvable!".format(puzzle_list),file = sys.stderr)
        #     exit(1)



    # @return puzzle_state The start state of the n-puzzle
    def start_state(self):
        return self.start_

    # @return puzzle_state The goal state of the n-puzzle
    def goal_state(self):
        return self.goal_

    # @return bool True if the puzzle is solvable
    def is_solvable(self):
        inversions = self.__get_inversion()
        x_row_bottom = self.width_ - self.start_state().x_index_//self.width_
        if self.width_%2 == 0: #even
            return x_row_bottom%2 != inversions%2
        else: #odd
            return inversions%2 == 0


    def __str__(self):
        return "{}-puzzle".format(self.size_)

    def __parse_puzzle(self, file):
        puzzle_list = []
        for i in range(0, self.width_):
            line = file.readline().strip().strip(",").split(",")
            if len(line) != self.width_:
                raise Exception("The width of puzzle line {} not equal to puzzle width".format(i))
            for char in line:
                if char.isnumeric():
                    num = int(char)
                    if num < 0 or num >= self.size_:
                        raise Exception("Number {} not in range 1~{}".format(num, self.size_ - 1))
                else:
                    num = "x"

                if num == 0:
                    num = "x"

                if num in puzzle_list:
                    raise Exception("You can't have two {} in one puzzle".format(num))
                puzzle_list.append(num)
        return puzzle_list


    def __get_inversion(self):
        count = 0
        for i in range(0,self.size_):
            if self.start_state().state_list_[i] =="x":
                continue
            for j in range(i+1,self.size_):
                if self.start_state().state_list_[j] == "x":
                    continue
                if self.start_state().state_list_[i] > self.start_state().state_list_[j]:
                    count += 1
        return count




