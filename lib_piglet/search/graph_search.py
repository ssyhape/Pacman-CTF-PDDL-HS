# search/graph_search.py
#
# Implements the Graph-Search algorithm:
# Given an expander and an open list, this approach will
# search until it finds the goal state.
#
# The expansion order is determined by the type of open list.
# The task environment is determined by the expander
#
# @author: mike
# @created: 2020-07-16
#
import sys, time
from lib_piglet.search.base_search import base_search
from lib_piglet.search.base_search import search_node
from lib_piglet.expanders.pddl_expander import pddl_expander


class graph_search(base_search):
    

    # Search the path between two state
    # @param start_state The start of the path
    # @param goal_state Then goal of the path
    # @return a list of locations between start and goal
    def get_path(self,start_state, goal_state):
        self.open_list_.clear()
        self.all_nodes_list_.clear()
        self.reset_statistic()
        self.start_ = start_state
        self.goal_ = goal_state
        self.start_time = time.process_time()
        start_node = self.generate(start_state, None, None)
        self.open_list_.push(start_node)
        self.all_nodes_list_[start_node] = start_node

        # continue while there are still nods on OPEN
        while (len(self.open_list_) > 0):
            current: search_node = self.open_list_.pop()
            current.close()
            self.nodes_expanded_ +=1

            # If have time_limit, break time out search.
            if self.time_limit_ < sys.maxsize:
                self.runtime_ = time.process_time() - self.start_time
                if self.runtime_ > self.time_limit_:
                    self.status_ = "Time out"
                    return None
            # goal example. if successful, return the solution

            if self.goal_test_function_(current.state_, goal_state):
                self.solution_ = self.solution(current)
                self.status_ = "Success"
                self.runtime_ = time.process_time() - self.start_time
                return self.solution_

            # expand the current node
            for succ in self.expander_.expand(current):
                # each successor is a (state, action) tuple which
                # which we map to a corresponding search_node and push
                # then push onto the OPEN list

                succ_node = self.generate(succ[0], succ[1], current)
                # succ_node not in any list, add it to open list
                if succ_node not in self.all_nodes_list_:
                    # we need this open_handle_ to update the node in open list in the future
                    succ_node.open_handle_ = self.open_list_.push(succ_node)
                    self.all_nodes_list_[succ_node] = succ_node
                    self.nodes_generated_+= 1

                # succ_node only have the same hash and state comparing with the on in the all nodes list
                # It's not the one in the all nodes list,  we need the real node in the all nodes list.
                exist = self.all_nodes_list_[succ_node]
                if not exist.is_closed():
                    self.relax(exist, succ_node)

        # OPEN list is exhausted and we did not find the goal
        # return failure instead of a solution
        self.runtime_ = time.process_time() - self.start_time
        self.status_ = "Failed"
        return None

    def relax(self, exist:search_node, new:search_node):
        if exist.g_ > new.g_:
            exist.f_ = new.f_
            exist.g_ = new.g_
            exist.depth_ = new.depth_
            exist.instance_ = new.instance_
            exist.action_ = new.action_
            exist.timestep_ = new.timestep_
            exist.h_ = new.h_
            exist.parent_ = new.parent_
            if exist.open_handle_ is not None:
                # If handle exist, we are using bin_heap. We need to tell bin_heap one element's value
                # is decreased. Bin_heap will update the heap to maintain priority structure.
                self.open_list_.decrease(exist.open_handle_)