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
from lib_piglet.solution.solution import solution


class dijkstra_search(base_search):

    # get distance from all state to target_state
    # @param target_state Then target_state of the search
    # @return a dictionary contains each state and distance from this state to target state
    def get_path(self,target_state):
        self.open_list_.clear()
        self.all_nodes_list_.clear()
        self.reset_statistic()
        self.max_depth_ = 0
        self.goal_ = target_state
        self.start_time = time.process_time()
        start_node = self.generate(target_state, None, None)
        self.open_list_.push(start_node)
        self.all_nodes_list_[start_node] = start_node

        # continue while there are still nods on OPEN
        while (len(self.open_list_) > 0):
            current: search_node = self.open_list_.pop()
            current.close()
            self.nodes_expanded_ +=1
            if current.depth_ > self.max_depth_:
                self.max_depth_ = current.depth_
            # If have time_limit, break time out search.
            if self.time_limit_ < sys.maxsize:
                self.runtime_ = time.process_time() - self.start_time
                if self.runtime_ > self.time_limit_:
                    self.status_ = "Time out"
                    return None

            if self.nodes_expanded_%100000 == 0:
                print(self.nodes_expanded_)

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

        # OPEN list is exhausted, dijkstra finish
        self.solution_ = self.solution()
        self.status_ = "Success"
        self.runtime_ = time.process_time() - self.start_time
        return self.solution_

    def relax(self, exist:search_node, new:search_node):
        if exist.g_ > new.g_:
            exist.f_ = new.f_
            exist.g_ = new.g_
            exist.h_ = new.h_
            exist.parent_ = new.parent_
            if exist.open_handle_ is not None:
                # If handle exist, we are using bin_heap. We need to tell bin_heap one element's value
                # is decreased. Bin_heap will update the heap to maintain priority structure.
                self.open_list_.decrease(exist.open_handle_)

    # extract the computed solution by following backpointers
    def solution(self):
        sol = {}
        for node in self.all_nodes_list_:
            sol[node.state_] = node.g_

        return solution(sol,self.max_depth_,self.nodes_expanded_)