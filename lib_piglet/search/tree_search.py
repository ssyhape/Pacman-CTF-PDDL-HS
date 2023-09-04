# search/tree_search.py
# 
# Implements the Tree-Search algorithm:
# Given an expander and an open list, this approach will 
# search until it finds the goal state.
# 
# The expansion order is determined by the type of open list.
# The task environment is determined by the expander
# 
# @author: dharabor
# @created: 2020-07-16
#
import time, sys
from lib_piglet.search.base_search import base_search
from lib_piglet.search.search_node import search_node


class tree_search(base_search):

    # Search the path between two state
    # @param start_state The start of the path
    # @param goal_state Then goal of the path
    # @return a list of locations between start and goal
    def get_path(self,start_state, goal_state, depth_limit:int = sys.maxsize, cost_limit:int  = sys.maxsize):
        self.open_list_.clear()
        self.reset_statistic()
        self.start_ = start_state
        self.goal_ = goal_state
        self.start_time = time.process_time()
        start_node = self.generate(start_state, None, None)
        self.open_list_.push(start_node)

        # For depth/cost limited search, we need to track the minimal threshold of nodes out of bound.
        min_next_d = sys.maxsize
        min_next_f = sys.maxsize

        # continue while there are still nods on OPEN
        while (len(self.open_list_) > 0):
            current: search_node = self.open_list_.pop()

            self.nodes_expanded_ +=1
            # If have time_limit, break time out search.
            if self.time_limit_ < sys.maxsize:
                self.runtime_ = time.process_time() - self.start_time
                if self.runtime_ > self.time_limit_:
                    self.status_ = "Time out"
                    return None,min_next_d,min_next_f

            # goal example. if successful, return the solution
            if self.goal_test_function_(current.state_, goal_state):
                self.solution_ = self.solution(current)
                self.status_ = "Success"
                self.runtime_ = time.process_time() - self.start_time
                return self.solution_,min_next_d,min_next_f

            # expand the current node
            for succ in self.expander_.expand(current):
                # each successor is a (state, action) tuple which
                # which we map to a corresponding search_node and push
                # then push onto the OPEN list
                succ_node = self.generate(succ[0], succ[1], current)
                # Check does child node exceed depth limit or cost limit
                if succ_node.depth_ > depth_limit or succ_node.f_ > cost_limit:
                    if min_next_d > succ_node.depth_:
                        min_next_d = succ_node.depth_
                    if min_next_f > succ_node.f_:
                        min_next_f = succ_node.f_
                    continue
                self.open_list_.push(succ_node)
                self.nodes_generated_+=1

        # OPEN list is exhausted and we did not find the goal
        # return failure instead of a solution
        self.runtime_ = time.process_time() - self.start_time
        self.status_ = "Failed"
        return None, min_next_d, min_next_f









