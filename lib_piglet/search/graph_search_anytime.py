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


class graph_search_anytime(base_search):
    
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
        self.UB = sys.maxsize
        self.first_solution_time_ = None
        self.re_expansions_ = 0
        self.solution_ = None

        # continue while there are still nodes on OPEN
        while (len(self.open_list_) > 0):
            current: search_node = self.open_list_.pop()
            current.close()
            current.open_handle_ = None
            self.nodes_expanded_ +=1

            # if current.expanded:
            #     self.re_expansions_ +=1
            # else:
            #     current.expanded=True

            

            # If have time_limit, break time out search.
            if self.time_limit_ < sys.maxsize:
                self.runtime_ = time.process_time() - self.start_time
                if self.runtime_ > self.time_limit_:
                    if self.solution_ == None:
                        self.status_ = "Time out"
                        return None
                    else:
                        # If out of time, but have current best, return suboptimal solution
                        self.status_ = "Suboptimal"
                        return self.solution_
            
            # update the upper bound if reach a goal node.
            if self.goal_test_function_(current.state_, goal_state):
                self.UB = current.g_
                self.solution_ = self.solution(current)
                if self.first_solution_time_ == None:
                    self.first_solution_time_ = time.process_time() - self.start_time
                continue

            # expand the current node
            for succ in self.expander_.expand(current):
                # each successor is a (state, action) tuple which
                # which we map to a corresponding search_node and push
                # then push onto the OPEN list
                succ_node = self.generate(succ[0], succ[1], current)

                if succ_node.g_ + succ_node.h_ > self.UB:
                    # Prune the node if unweighted f is larger than upper bound.
                    continue

                # succ_node not in any list, add it to open list
                if succ_node not in self.all_nodes_list_:
                    # we need this open_handle_ to update the node in open list in the future
                    succ_node.open_handle_ = self.open_list_.push(succ_node)
                    self.all_nodes_list_[succ_node] = succ_node
                    self.nodes_generated_+= 1
                else:
                    # succ_node only have the same hash and state comparing with the on in the all nodes list
                    # It's not the one in the all nodes list,  we need the real node in the all nodes list.
                    exist = self.all_nodes_list_[succ_node]
                    self.relax(exist, succ_node)

        # OPEN list is exhausted if nodes 
        if self.solution_ == None:
            self.status_ = "Failed"
            return None
            
        self.runtime_ = time.process_time() - self.start_time
        self.status_ = "Optimal"
        return self.solution_

    def relax(self, exist:search_node, new:search_node):
        if new.g_ < exist.g_:
            exist.f_ = new.f_
            exist.g_ = new.g_
            exist.h_ = new.h_
            exist.depth_ = new.depth_

            exist.parent_ = new.parent_

            if exist.is_closed():
                # move the closed node into open list and mark open
                exist.open_handle_ = self.open_list_.push(exist)
                exist.open()
                self.re_expansions_ +=1
            elif exist.open_handle_ is not None:
                # If handle exist, we are using bin_heap. We need to tell bin_heap one element's value
                # is decreased. Bin_heap will update the heap to maintain priority structure.
                self.open_list_.decrease(exist.open_handle_)
    
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
                self.re_expansions_,
                None if self.first_solution_time_ == None else round(self.first_solution_time_,4),
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
        self.first_solution_time_ = None
        self.re_expansions_ = 0
        self.UB = sys.maxsize