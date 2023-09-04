# expander/grid_expander.py
#
# Expand function for graph
#
# Given a current search node, the expander checks the set of valid graph actions
# and generates search node successors for each.
#
# @author: mike
# @created: 2020-07-27
#
from lib_piglet.expanders.base_expander import base_expander
from lib_piglet.domains.graph import graph, vertex
from lib_piglet.search.search_node import search_node


class graph_action:
    
    def __init__(self, action: int, cost:int):
        self.move_: int = action
        self.cost_: int = cost

class graph_expander(base_expander):

    def __init__(self,g: graph ):
        self.domain_: graph = g
        self.succ_: list = []

    def expand(self, current_node: search_node):
        self.succ_.clear()
        current_vertex: vertex = current_node.state_
        for v, cost in current_vertex.get_connections():
            successor = v
            action = graph_action(v.get_id(),cost)
            self.succ_.append((successor, action))
        return self.succ_[:]

    def __str__(self):
        return self.domain_.domain_file_


