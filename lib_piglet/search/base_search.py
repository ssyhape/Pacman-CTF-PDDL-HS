# search/base_search.py
# This module defines a base search class and what attribute/method a search class should have.
#
# @author: mike
# @created: 2020-07-15
#
import sys
from lib_piglet.expanders.base_expander import base_expander
from lib_piglet.search.search_node import search_node
from lib_piglet.solution.solution import solution
from lib_piglet.cli.cli_tool import statistic_template,statistic_header
from typing import Callable


class base_search:


    def __init__(self, open_list, expander:base_expander, heuristic_function = None, time_limit: int = sys.maxsize):
        self.open_list_ = open_list
        self.expander_: base_expander = expander
        self.time_limit_ = time_limit
        self.heuristic_function_: Callable = heuristic_function
        self.goal_test_function_ : Callable= self.expander_.domain_.is_goal
        self.all_nodes_list_ = {}

        self.nodes_generated_: int = 0
        self.nodes_expanded_: int = 0
        self.runtime_: float = 0
        self.start_time_: float = 0
        self.solution_: solution = None
        self.start_: object = None
        self.goal_: object = None
        self.status_: str = None
        self.heuristic_weight_:float = 1.0
        self.max_depth_ = 0

    # Search the path between two state
    # @param start_state The start of the path
    # @param goal_state Then goal of the path
    # @return a list of locations between start and goal
    def get_path(self,start_state, goal_state):
        raise NotImplementedError()

    # Generate search_node objects for a given state
    # For this operatin we we need to know:
    # @param state: the state which the search node maps to
    # @param action: the action which generated the state (could be [None])
    # @param parent: the parent state (could be [None])
    def generate(self, state, action, parent: search_node):

        retval = search_node()
        retval.state_ = state
        retval.action_ = action
        if (parent == None):
            # initialise the node from scratch
            # NB: we usually do this only for the start node
            retval.g_ = 0
            retval.depth_ = 0
            retval.timestep_ = 0
        else:
            # initialise the node based on its parent
            retval.g_ = parent.g_ + action.cost_
            retval.depth_ = parent.depth_ + 1
            retval.parent_ = parent
            retval.timestep_= parent.timestep_ + 1


        if self.heuristic_function_ is None:
            retval.h_ = 0
            retval.f_ = retval.g_
        else:
            retval.h_ = self.heuristic_function_(self.expander_.domain_,retval.state_, self.goal_)
            retval.f_ = retval.g_ + retval.h_ * self.heuristic_weight_
        return retval

    # extract the computed solution by following backpointers
    def solution(self, goal_node: search_node):
        tmp = goal_node
        depth = goal_node.depth_
        cost = goal_node.g_
        sol = []
        while (tmp != None):
            sol.append(tmp)
            tmp = tmp.parent_

        sol.reverse()
        return solution(sol,depth,cost)

    # Get statistic information
    # @return list A list of Statistic information
    def get_statistic(self):
        sta_ = [self.status_]
        if self.solution_!=None:
            sta_ += self.solution_.get_solution_info()
        else:
            sta_ += [None, None]
        sta_ += [self.nodes_expanded_,
                self.nodes_generated_,
                round(self.runtime_,4),
                self.start_,
                self.goal_,
                self.expander_]

        return sta_

    def reset_statistic(self):
        self.nodes_generated_: int = 0
        self.nodes_expanded_: int = 0
        self.runtime_: float = 0
        self.start_time_: float = 0
        self.solution_: solution = None
        self.start_ = None
        self.goal_ = None






